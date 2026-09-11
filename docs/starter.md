# Starter commands

This draft adds `bloom new`, `bloom run`, `bloom run --web` and `bloom build`.
They are not available in the stable npm 0.4.16 package. To try the draft from
its checkout, first create a package archive:

```sh
npm pack --ignore-scripts
npm exec --package ./bloomengine-engine-0.4.16.tgz -- bloom new my-game
cd my-game
npm start
```

The creation command installs dependencies and writes one TypeScript source,
an asset example, Bloom/Perry version metadata and the native-library permissions.
It packs the exact engine package running the command into `.bloom/engine.tgz`,
which the project references locally. This keeps preview and registry packages
with the same version number from being confused. Keep the archive and generated
lockfile with the project. `--engine-package <local.tgz>` supplies a specific
compatible archive; `--no-install` defers dependency installation.

## Prerequisites and targets

- Node.js 18+ with npm, plus Rust.
- Windows native: Python 3.12+, Git and Visual Studio C++ build tools. Use a
  developer shell. The first build prepares Perry 0.5.1220 and matching source
  libraries under `%LOCALAPPDATA%/Bloom/perry-0.5.1220`. The existing verified
  setup helper handles the compiler download and Cargo commands. Later builds
  verify and reuse that cache.
- macOS/Linux native: Perry 0.5.1220 and the host platform SDK. These starter
  command paths require their own runtime qualification; local evidence is Windows.
- Web: Perry 0.5.1220, wasm-pack and a WebGPU-capable browser. Set BLOOM_PERRY to
  the executable prepared by Windows if that compiler is not on PATH.

```sh
npm run web
npm exec -- bloom run --web --port 8081
npm exec -- bloom build --target web
npm run build
```

Web run builds and serves at `http://127.0.0.1:8080` until Ctrl+C. It does not
automatically open a browser. Both native and web build the same `main.ts`.
The web asset manifest is generated before serving, so the glue prefetches the
template's text asset before synchronous game initialization. Native run uses
the distribution directory as its working directory to find the copied assets.

Build targets are `native`, `web`, `windows`, `linux` and `macos`; native targets
must match the current host. Mobile deployment remains in its platform SDK flow.
Release builds omit Perry's debug-symbol flag. Signing, installers, DXC/DXIL
distribution and publishing are separate packaging work.

The Windows toolchain location can be changed with BLOOM_TOOLCHAIN_DIR. To use
an explicitly prepared native profile, set BLOOM_PERRY, PERRY_RUNTIME_DIR and
PERRY_WORKSPACE_ROOT together. BLOOM_PYTHON controls the setup interpreter.
Perry currently requires crate-local native Cargo output; the command rejects
a configured CARGO_TARGET_DIR instead of reporting a successful build with no
executable. General long Windows project paths remain unqualified.

## Development and validation

The template uses `runGameLifecycle` on both targets, with init, 60 Hz fixed
update, variable update, interpolated draw and cleanup. See the
[game-loop contract](game-loop.md). Edit, stop and rerun to rebuild; automatic
hot reload and device-loss recovery are not implemented by this command.

The original #171 installed-package acceptance creates the project through the installed
npm command, builds the unmodified template natively, and completes the full
web build from that same source. A bounded copy adds capture/state/cleanup
observations for native execution. It renders the 800x450 greeting and square,
loads `assets/welcome.txt` and cleans up once on Radeon DX12. This does not prove
visible presentation, the new fixed-lifecycle template, compiled-starter browser rendering or a clean machine's
shader DLL packaging.

The installed-package CI check verifies default creation from the exact packed
engine and rejects missing compilers and unsupported targets. Unit tests cover
manifest drift, existing-project preservation, asset inventory and the local
server. See [the retained acceptance report](evidence/windows-starter-cli-v1.md).

The fixed-lifecycle follow-up repeats fresh installed default creation, native
build and the bounded greeting/asset/cleanup run using the revised template,
then completes its full web build. See [the lifecycle evidence](evidence/fixed-game-lifecycle-v1.md).
The browser runtime still needs to qualify this complete starter.

The [complete starter browser gate](evidence/installed-starter-browser-v1.md)
now runs the installed web command and retains its unchanged generated website
for hosted rendering, including the actual asset read, text draw and cleanup.
It also covers the production bridge for named void callbacks in Perry WASM.
Its hosted result is pending.
