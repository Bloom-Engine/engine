# My Bloom game

This project pins Bloom in `package.json` and the compatible Perry version in
`bloom.json`. `.bloom/engine.tgz` contains the exact engine package used by the
creation command, so a preview cannot silently install an older registry package
with the same version number. Keep that archive and `package-lock.json` with the
project after dependency installation.

Install Node.js 18+ and Rust. Web builds also require wasm-pack and Perry
0.5.1220 on PATH, or an executable path in BLOOM_PERRY. Windows native builds
require Python 3.12+, Git and Visual Studio C++ build tools, and automatically
prepare a matching Perry compiler/runtime under your local application data.
Run from a Visual Studio developer shell so the native linker is available.
macOS/Linux native builds require Perry 0.5.1220 and their normal platform SDK.

```sh
npm install
npm start
npm run web
npm run build
```

`npm start` builds and runs natively. `npm run web` builds the same `main.ts` and
serves it at http://127.0.0.1:8080; open that address in a WebGPU-capable browser.
Ctrl+C stops the server. `npm exec -- bloom run --web --port 8081` changes the port.
Build-only web output is available with `npm exec -- bloom build --target web`.
Native builds target the current host; other native/mobile targets require their
platform SDK and packaging flow. Release builds omit Perry's debug-symbol flag;
this command does not sign, install or publish a distributable release.

Edit `main.ts`, stop and rerun the command to rebuild. There is no automatic hot
reload yet. The template has init, variable update, draw and cleanup functions.
Cleanup runs after the final frame during normal shutdown. Focus loss does not
automatically pause the game; the template caps its variable delta at 0.1 seconds.
Fixed update and recovery after device loss are separate engine work.

`assets/welcome.txt` demonstrates loading the same asset on native and web. Asset
paths are relative to the project and are copied beside the generated binary or
web page. Keep runtime resource disposal in `cleanup`. Do not put another
beginDrawing/endDrawing pair inside the runGame callback.

Missing prerequisites, incompatible versions and manifest errors stop the build.
Native failures retain the compiler's output; browser startup failures appear in
the browser console. The first native build compiles the engine and can take
several minutes. Later builds reuse Cargo and Perry caches. Current Windows
native builds require short project paths and crate-local Cargo output; unset
CARGO_TARGET_DIR if it is configured globally.
