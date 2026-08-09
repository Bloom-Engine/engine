# Issue #135 temporal cost ledger

This ledger consolidates the timing and storage cost of every production
history or mask added while implementing issue #135. It also records the
capture-only diagnostic masks so their temporary cost cannot be mistaken for
normal-frame renderer memory.

The qualified renderer revision is `29abf80` on Apple M1 Max / Metal. The
performance harness changes used to select the retained-transparency workload
and pass profiler do not modify the renderer.

## Production history and mask inventory

| Change | Added storage | Added production passes/draws | Qualification |
|---|---:|---:|---|
| SSR validity and firefly bound | 0 B | 0 / 0 | Reuses the two existing HDR histories and temporal pass |
| SSGI validity and finite seeding | 0 B of history | 0 / 0 | Reuses the two existing 3D histories |
| SSGI cached diffuse probe value | 32,640 B at native 1080p | 0 / 0 | Adds 16 B/probe to the existing header and removes a resolve texture lookup |
| SSGI geometric continuity | 65,280 B at native 1080p | 0 / 0 | Adds prior world position/normal to the probe header; no texture, bind group, dispatch, or draw |
| TAA validity and ownership | 0 B | 0 / 0 | Reuses the two existing TAA histories |
| Auto-exposure validity | 0 B | 0 / 0 | Reuses the existing two 1x1 histories |
| PT reset/reseed contract | 0 B | 0 / 0 | Reuses compatible SVGF histories; no retained bytes can seed a zero-sample epoch |
| Cached-model motion | 576 B CPU for the measured one-instance stream | 0 / 0 | Reuses `prev_mvp`, the velocity MRT, and the existing draw |
| Immediate-primitive motion | 1,728 B CPU for the measured three-primitive stream | 0 / 0 | Reuses the vertex tangent lane, upload, velocity MRT, and draw |
| Unkeyed two-joint skin motion | 448 B CPU | 0 / 0 | Reuses the existing current/previous joint buffers and draw |
| Keyed/legacy skin and foliage motion | 0 added GPU storage | 0 / 0 | Reuses existing palette buffers, velocity MRT, uploads, and draw |
| Reactive transparency coverage | 1 B/render pixel while active | 0 / 0 | One transient R8 attachment on the existing translucent pass; absent otherwise |
| Camera-cut/reset API and validity flags | No dynamic allocation | 0 / 0 | Invalidates compatible storage in place |

The active mask is 1,166,400 bytes at the shipped Medium default
(1440x810 render extent on a 1920x1080 surface) and 2,073,600 bytes at native
Ultra. Ordinary opaque, TAA-disabled, and non-reactive-translucency frames do
not materialize it. After warm-up, both mask states report zero render-graph
compiles, zero transient physical creations, and zero bind-group creations
per frame.

## Active-mask timing

The qualification tool renders the same retained glTF BLEND cube in both
states. `BLOOM_TEMPORAL_REACTIVE=off` is the feature-off control;
`BLOOM_TEMPORAL_REACTIVE=on` selects the production R8 target and reactive TAA
pipeline. Four paired runs used 300 warm-up and 900 measured uncapped frames
at 1920x1080.

At the shipped Medium default:

| Full-frame CPU metric | Feature off | Active mask | Change |
|---|---:|---:|---:|
| Mean | 2.3292 ms | 2.2716 ms | -2.47% |
| P50 | 2.4405 ms | 2.4556 ms | +0.0150 ms / +0.62% |
| P95 | 3.5403 ms | 3.0651 ms | -13.42% |
| Preparation mean | 0.0602 ms | 0.0595 ms | -0.0007 ms |
| Compiled graph passes | 15 | 15 | 0 |

The P50 delta is below run-to-run scheduling noise and there is no measurable
default-frame regression. Ultra wall-submit runs were scheduling-unstable:
their feature-off/active median means were 4.2031/4.7827 ms while individual
P95 values ranged from 4.9991 to 19.6675 ms. A dedicated 120-frame
timestamped pass run therefore isolated the affected work:

| Ultra GPU pass mean | Feature off | Active mask |
|---|---:|---:|
| Existing translucent pass | 1.6158 ms | 1.4295 ms |
| Existing TAA pass | 1.9063 ms | 1.8727 ms |

Metal timestamp noise means the lower active values are not claimed as an
optimization. They do show no localized pass regression and confirm that no
pass was added. The profiler deliberately blocks for timestamp readback, so
its wall-frame time is not used as a performance result.

The earlier frozen-revision producer comparisons remain the before/after code
regression gates:

| Affected path | Before median full-frame mean | After | Uploads |
|---|---:|---:|---:|
| Immediate primitives | 2.2396 ms | 2.2360 ms (-0.16%) | 23,264 B/frame unchanged |
| Custom reactive authoring, ordinary path | 2.6678 ms | 2.4082 ms (-9.73%; noisy, not claimed) | 23,264 B/frame unchanged |
| Legacy/unkeyed skin and producer audit | 2.2421 ms | 2.2273 ms (-0.66%) | 23,264 B/frame unchanged |

## Capture-only diagnostic memory

The following native-1080p telemetry is descriptive, not steady-state cost.
Each diagnostic family is created only after a capture request, records one
extra pass, and is released after readback.

| Diagnostic family | Temporary textures | Temporary readback | Persistent | Capture passes |
|---|---:|---:|---:|---:|
| TAA | 33,177,600 B | 33,177,600 B | 0 B | 1 |
| SSR | 1,036,800 B | 3,179,520 B | 0 B | 1 |
| SSGI | 1,044,480 B | 1,114,112 B | 0 B | 1 |
| Realtime PT | 8,294,400 B | 8,294,400 B | 0 B | 1 |

All four telemetry groups reported `resources_live=false` after capture. Raw
SSR/SSGI/HDR copies reuse production textures and add no diagnostic render
pass by themselves.

## Regression qualification

The full quick lane passes on the ledger/tooling tree:

- 325 unit tests passed, 1 intentionally ignored;
- 57 runnable GPU goldens passed, 2 intentionally ignored;
- all 4 render-target integration tests passed;
- strict clippy/format, all-platform FFI/schema parity, web arity, Wasm
  compilation, quality governance, visual fault controls, and canonical
  examples passed.

The qualification tool is the only executable changed by this checkpoint.
Production renderer shaders, resources, passes, and accepted images are
unchanged.

## Commands

```sh
BLOOM_TEMPORAL_REACTIVE=<off|on> \
BLOOM_RENDER_PERF_ENGINE_REVISION=<revision> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 300 --frames 900 \
  --quality-preset 2 --reactive-transparency --out <report.json>

BLOOM_TEMPORAL_REACTIVE=<off|on> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 180 --frames 120 \
  --quality-preset 4 --reactive-transparency --profile-passes \
  --out <profile-report.json>

cargo test --manifest-path native/shared/Cargo.toml --test golden_render \
  cached_alpha_tested_card_motion_writes_velocity_and_bounds_trails \
  -- --nocapture
```
