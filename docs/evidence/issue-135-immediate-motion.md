# Issue #135 immediate-primitive motion evidence

This qualification covers raylib-style immediate 3D submissions before and
after adding previous-position ownership. The comparison base is
`5d5d9a1e4e15b991111984a8cefd5d9b1307d15e`.

## Motion sequence

The headless Metal test warms a static pose for eight frames, translates an
immediate cube from x=-1.6 to x=1.6, captures the first moved frame's temporal
diagnostics, and evaluates 24 output frames:

- 7,234 pixels carry meaningful motion;
- the visible move is strong (`movement_mean = 15.8512`);
- severe trail duration is zero frames;
- coherent frame-four outliers are 0.7599%, below the 2% gate;
- settled jitter-cycle flicker is 0.4695, below the 2.0 gate.

First appearances, primitive-kind changes, vertex-count changes, empty
intervening frames, and explicit temporal resets have unit-tested
previous=current seeding. They cannot inherit an unrelated slot's motion.

## Performance and memory

The controlled `tools/render-perf` scene is the affected path: one immediate
plane and cube, Ultra preset, 40 point lights, 300 warm-up plus 900 measured
frames at 1920x1080 on Apple M1 Max / Metal. Values are medians of three runs
per revision.

| CPU metric | Before | After | Change |
|---|---:|---:|---:|
| Full render-submit mean | 2.2396 ms | 2.2360 ms | -0.16% |
| Full render-submit P50 | 2.3505 ms | 2.3855 ms | +1.49% |
| Full render-submit P95 | 2.5876 ms | 2.6064 ms | +0.72% |
| Full render-submit P99 | 3.7760 ms | 3.6173 ms | -4.20% |
| Submission preparation mean | 0.0341 ms | 0.0387 ms | +0.0045 ms |

An immediate alternating fourth pair moved from 2.2342 to 2.2199 ms mean;
the small percentile changes above are below run-to-run scheduling variance,
while the complete-frame mean shows no regression.

The affected steady upload remains exactly 23,264 bytes per frame. There is no
vertex-stride change, texture upload, GPU allocation, bind group, draw, or
render pass. The three-primitive motion test retains 1,728 bytes of grow-only
CPU capacity. Runtime telemetry reports the live entry count and capacity,
plus zero GPU bytes and zero added passes.

## Commands

```sh
BLOOM_RENDER_PERF_ENGINE_REVISION=<revision> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 300 --frames 900 --out <report.json>

cargo test --manifest-path native/shared/Cargo.toml --test golden_render \
  immediate_primitive_motion_writes_velocity_and_bounds_trails -- --nocapture
```
