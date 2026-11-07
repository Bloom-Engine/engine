# Issue #135 remaining motion-producer evidence

This qualification closes the engine-owned motion-producer audit after the
retained-model, immediate-primitive, and custom-particle slices. The comparison
base is `6921e4146fa46f09f57ddb314f3dbed56ab8c07e`; the qualified implementation
is `53ce363cf33f7ddd305f078d7a7f95bd54b820d5`.

## Visual sequences

The native Metal corpus runs the production velocity MRT and TAA resolve:

- legacy immediate skinning writes 10,214 meaningful motion pixels, leaves
  zero severe trail frames, and has 0.8713% frame-four outliers;
- the unkeyed editor skin API also writes 10,214 meaningful motion pixels,
  leaves zero severe trail frames, and has 0.8636% frame-four outliers;
- after an empty intervening frame, that unkeyed skin writes zero stale-motion
  pixels on reappearance;
- procedural foliage wind writes 7,348 meaningful prior-deformation pixels;
- static decal spawn and expiry both leave zero severe trail frames, with
  0.6393% and 0.5157% frame-four outliers respectively;
- a procedural cloud-field discontinuity leaves zero severe trail frames and
  1.1368% frame-four outliers.

Every moving-geometry sequence captures `taa-motion.png` and asserts a visible
negative control. The decal and cloud tests deliberately qualify their
zero-velocity contracts through rejection/recovery instead of inventing
geometric motion for radiance or lifecycle changes.

## Performance and memory

Four frozen before/after binaries ran in alternating order with 300 warm-up and
900 measured frames at 1920x1080 on Apple M1 Max / Metal. The median
full-frame mean changed from 2.2421 ms to 2.2273 ms (-0.66%). All four paired
means were lower after the change. Median P50 changed from 2.4567 ms to
2.4696 ms (+0.53%) and median P95 from 2.7256 ms to 2.7527 ms (+0.99%), both
inside the run-to-run scheduling range; the deliberately noisy first pair was
also the largest improvement.

Separate 32-frame traces are exact: 23,264 upload bytes per frame before and
after. The controlled scene adds no GPU allocation, pass, draw, bind, or
upload. Active legacy skinning reuses the existing previous-joint buffer,
binding, upload, and velocity attachment; only weighted vertices execute the
previous-palette transform already used by retained skinned geometry.

The unkeyed API adds two grow-on-demand CPU palette streams and reuses their
inner allocations after warm-up. One active two-joint skin retains 448 bytes
of measured CPU capacity. Telemetry reports this capacity and confirms zero
added GPU bytes and passes.

## Commands

```sh
cargo test --manifest-path native/shared/Cargo.toml \
  --test golden_render motion_producer_audit -- --nocapture

cargo test --manifest-path native/shared/Cargo.toml \
  --test golden_render legacy_skinned_motion_uses_staged_previous_palette_and_bounds_trails \
  -- --nocapture

BLOOM_RENDER_PERF_ENGINE_REVISION=<revision> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 300 --frames 900 --out <report.json>

BLOOM_RENDER_PERF_ENGINE_REVISION=<revision> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 32 --frames 32 \
  --trace-dir <trace-directory> --out <upload-report.json>
```
