# @bloomengine/jolt-prebuilt

Prebuilt JoltPhysics + bloom_jolt static libraries for Bloom Engine's published
target matrix. Targets without a published archive use the engine package's C++
source fallback.

## Why this exists

`@bloomengine/engine` consumes JoltPhysics for its rigid/soft body physics. Compiling Jolt from C++ via cmake on a consumer's first build takes 5–15 minutes cold. This package ships the libraries prebuilt so consumers skip that step entirely.

## Layout

```
lib/
  macos-arm64/      Apple Silicon
  macos-x64/        Intel Mac
  ios-arm64/        iOS device
  ios-arm64-sim/    iOS simulator on Apple Silicon
  ios-x64-sim/      iOS simulator on Intel
  tvos-arm64/       tvOS device
  tvos-arm64-sim/
  tvos-x64-sim/
  watchos-arm64/    watchOS device
  watchos-arm64-sim/
  watchos-x64-sim/
  linux-x64/
  linux-arm64/
  win32-x64/
  android-arm64/
  android-armv7/
  android-x64/
```

Each variant directory contains `libJolt.a` (or `Jolt.lib` on Windows) and `libbloom_jolt.a`.

## How `@bloomengine/engine` finds it

The engine's `native/shared/build.rs` walks up from `CARGO_MANIFEST_DIR` looking
for `node_modules/@bloomengine/jolt-prebuilt/lib/<os>-<arch>/`. If found, it
links the prebuilt archives and skips cmake entirely. If not found (or the env
var `BLOOM_JOLT_FROM_SOURCE=1` is set), it builds Jolt from the C++ source
bundled in `@bloomengine/engine`.

## Build / publish

Built by `.github/workflows/release.yml` on each tag push — a matrix job per platform produces the libraries on the appropriate native runner (`macos-14` for Apple targets, `ubuntu-22.04` for Linux/Android, `windows-latest` for Windows) and uploads them as artifacts. A final assembly job collects every artifact into this package's `lib/` tree and publishes via OIDC trusted publishing.

This package is versioned independently. `@bloomengine/engine` pins the
compatible prebuilt version in its dependencies.
