# @bloomengine/engine-prebuilt

Prebuilt `bloom_<platform>` static libraries for every (target) `@bloomengine/engine` supports.

## Why this exists

`@bloomengine/engine`'s per-platform Rust crates (`native/macos/`, `native/ios/`, `native/tvos/`, `native/windows/`, `native/linux/`, `native/android/`) each pull in heavy native dependencies — `aws-lc-sys`, `oboe-sys`, `minimp3-sys`, `metal`, `wgpu`, `objc2-*`, etc. — that need a target-specific C/C++ toolchain to compile from source.

Build hubs (e.g. `hub.perryts.com`) and CI runners that consume Bloom shouldn't have to host every host-platform toolchain just to build one engine. This package ships the resulting static archive prebuilt on the appropriate native runner so the consumer's link step finds a ready-to-link `.a` (or `.lib`).

Mirrors the [`@bloomengine/jolt-prebuilt`](../jolt-prebuilt) pattern.

## Layout

```
lib/
  aarch64-apple-darwin/      libbloom_macos_bundled.a
  aarch64-apple-ios/         libbloom_ios_bundled.a
  aarch64-apple-tvos/        libbloom_tvos_bundled.a
  aarch64-linux-android/     libbloom_android_bundled.a
  x86_64-pc-windows-msvc/    bloom_windows_bundled.lib
  x86_64-unknown-linux-gnu/  libbloom_linux_bundled.a
```

Each archive is **self-contained**: the Rust object code, JoltPhysics (`libJolt.a`), and the `bloom_jolt` C shim are merged into a single static archive via the per-platform bundler (`libtool -static` on Apple targets, `ar`-extract-then-merge on Linux/Android, `lib.exe` on Windows). Perry's link step only needs to point at one file.

The corresponding directory names are Rust target triples — Perry's `prebuilt:` resolution walks `node_modules` and picks the right one based on the build target.

## How `@bloomengine/engine` finds it

The engine's `package.json` declares per-target `prebuilt:` paths inside `perry.nativeLibrary.targets`:

```jsonc
"targets": {
  "macos-arm64": {
    "prebuilt": "@bloomengine/engine-prebuilt/lib/aarch64-apple-darwin/libbloom_macos_bundled.a",
    "frameworks": ["Metal", "QuartzCore", "AppKit", ...],
    "libs": ["c++"]
  },
  // bare `macos` entry stays as `crate:` fallback for in-repo dev
}
```

Perry's prebuilt resolution (issue #860) walks node_modules to find this package, then links the archive directly — no cargo, no rustc, no cc-rs invocation against engine source on the consumer side.

## Build / publish

Built by `.github/workflows/release.yml` on each tag push — a matrix job per target produces the library on the appropriate native runner (`macos-14` for Apple, `ubuntu-22.04` for Linux/Android, `windows-latest` for Windows) and uploads it as an artifact. A final assembly job collects every artifact into this package's `lib/` tree and publishes via OIDC trusted publishing.

The published version always matches the corresponding `@bloomengine/engine` version they were built against.
