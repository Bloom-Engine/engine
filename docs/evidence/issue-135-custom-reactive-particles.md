# Issue #135 custom-reactive particle evidence

This qualification covers the optional custom-translucency
`fs_reactive -> ReactiveTranslucentOut` contract. The comparison base is the
immediately preceding renderer checkpoint,
`80ce85ce2ce361636fd00bc0670b64edb9dbfcb6`.

## Motion sequence

The headless Metal test uses the production cached mesh, instanced material,
additive bucket, sorted custom-translucency pass, temporal-reactive target, and
TAA resolve. It warms one particle position for eight frames, teleports the
instance from x=-1.35 to x=1.35, captures the first moved frame's rejection
map, and evaluates 24 output frames:

- 2,222 pixels are classified as authored reactive rejection;
- the visible move is strong (`movement_mean = 5.5887`);
- severe trail duration is zero frames;
- coherent frame-four outliers are 1.0361%, below the 2% gate;
- settled jitter-cycle flicker is 0.4743, below the 2.0 gate.

The sequence then repeats with the same shader and motion but without
`fs_reactive`. Its reactive topology remains inactive, and its existing
color/depth/neighborhood rejection also meets the same recovery bounds. A
low-level two-material GPU test separately proves that an ordinary shader's
attachment-compatible sibling has an empty R8 write mask while an opt-in
shader unions its authored coverage.

Malformed `fs_reactive` output locations or types fail at material compile
time. A same-named helper function is not treated as an entry point.

## Performance and memory

The ordinary controlled `tools/render-perf` scene contains no custom
translucency. It therefore exercises the unchanged topology and proves the
new command scan is short-circuited. Four frozen before/after binaries ran in
alternating order with 300 warm-up and 900 measured frames at 1920x1080 on
Apple M1 Max / Metal. Values below are medians of four runs per revision.

| CPU metric | Before | After | Change |
|---|---:|---:|---:|
| Full render-submit mean | 2.6678 ms | 2.4082 ms | -9.73% |
| Full render-submit P50 | 2.6856 ms | 2.5938 ms | -3.42% |
| Full render-submit P95 | 4.5943 ms | 3.6891 ms | -19.70% |
| Full render-submit P99 | 6.6183 ms | 4.9526 ms | -25.17% |
| Submission preparation mean | 0.0782 ms | 0.0621 ms | -20.63% |

Host scheduling was noisy and trended downward across the alternating pairs,
so the lower after values are not claimed as an optimization. They do rule out
a measurable regression: every paired full-frame mean was lower after the
change, and the affected no-custom-material route performs no command scan.

Traced uploads remain exactly 23,264 bytes per frame before and after.
Ordinary custom-translucency frames add no image, pass, draw, upload, or
pipeline. An active opt-in frame adds one transient R8 image
(one byte per render pixel; 2,073,600 bytes at 1920x1080) and one lazily
compiled material sibling. Coverage is a second attachment on the existing
translucent pass, so added passes and draws remain zero.

## Commands

```sh
cargo test --manifest-path native/shared/Cargo.toml \
  instanced_particle_reactive_opt_in_bounds_trails_without_taxing_opt_out \
  --test golden_render -- --nocapture

BLOOM_RENDER_PERF_ENGINE_REVISION=<revision> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 300 --frames 900 --out <report.json>

BLOOM_RENDER_PERF_ENGINE_REVISION=<revision> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 32 --frames 32 \
  --trace-dir <trace-directory> --out <upload-report.json>
```
