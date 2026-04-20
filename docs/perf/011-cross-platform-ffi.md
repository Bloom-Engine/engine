# 011 — Port quality/profiler FFI to iOS/Win/Lin/Android/tvOS/web

**Effort:** ~1 day · **Expected gain:** Unblocks non-macOS TS API · **Status:** open

## Problem

The `QualityPreset` / `setShadowsEnabled` / `setProfilerEnabled` / etc. TS
API all work on macOS because the FFI wrappers (`bloom_set_quality_preset`,
`bloom_set_profiler_enabled`, …) only exist in `native/macos/src/lib.rs`.

Games running on iOS, Windows, Linux, Android, tvOS, or web will link but
fail at runtime (missing symbol) when those functions are called.

## The FFI surface that needs mirroring

Added in commit `95da6af` to macOS only. Look at the block near line ~1100 of
`native/macos/src/lib.rs` for the complete set. Summary:

```
bloom_set_quality_preset(preset: f64)
bloom_set_shadows_enabled(on: f64)
bloom_set_bloom_enabled(on: f64)
bloom_set_ssao_enabled(on: f64)
bloom_set_ssr_enabled(on: f64)
bloom_set_motion_blur_enabled(on: f64)
bloom_set_sss_enabled(on: f64)
bloom_set_profiler_enabled(on: f64)
bloom_get_profiler_frame_cpu_us() -> f64
bloom_get_profiler_frame_gpu_us() -> f64
bloom_print_profiler_summary()
```

Each one is a thin wrapper around an existing method on `engine().renderer`
or `engine().profiler`. Shared code (`native/shared/src/renderer.rs`,
`native/shared/src/profiler.rs`) is platform-agnostic and already exposes
everything.

## Proposed approach

1. Copy the FFI block from `native/macos/src/lib.rs` into each platform's
   `lib.rs` (`ios`, `tvos`, `windows`, `linux`, `android`).
2. For `native/web/src/lib.rs`: same functions with `#[wasm_bindgen]` instead
   of `#[no_mangle] extern "C"`. Follow the pattern of existing `_enabled`
   setters in the web crate.
3. `package.json` already declares the functions — no changes needed there.
4. Verify with `cargo check` on each crate (see CLAUDE.md for per-platform
   build commands).
5. For web: also update `native/web/index.html`'s FFI bridge if it explicitly
   lists functions (check `__perry` / `ffi` namespace mapping).

### Platform-specific adapter for TIMESTAMP_QUERY

Only macOS requests `wgpu::Features::TIMESTAMP_QUERY` at device creation
right now (see `native/macos/src/lib.rs` `request_device` call). Mirror that
conditional feature request in every platform so the profiler's GPU path
works elsewhere when the adapter supports it.

## Acceptance

- `cargo check` passes on all targets with the new FFI block.
- On iOS, a minimal test TS program calling `setQualityPreset(QualityPreset.Low)`
  doesn't crash and the shadows visibly disappear.
- Web build path: functions are callable from the JS glue layer; test in a
  browser with WebGPU enabled.

## Notes for the implementer

- Keep the FFI signatures identical across platforms — Perry's native library
  manifest in `package.json` is a single source of truth.
- Don't add the `print_profiler_summary` to web (stdout doesn't exist in
  browser). Return the summary string via NaN-boxed encoding instead.
- The web crate's `_str` and `_bytes` variants pattern applies if any future
  profiler function needs a string arg (none currently do).

## Files likely to change

- `native/ios/src/lib.rs`, `native/tvos/src/lib.rs`,
  `native/windows/src/lib.rs`, `native/linux/src/lib.rs`,
  `native/android/src/lib.rs`, `native/web/src/lib.rs`.
- All platform `lib.rs` files that do `request_device` — add the
  TIMESTAMP_QUERY conditional feature request.
