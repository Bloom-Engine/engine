# Issue #56 coherent quality-preset evidence

This qualification compares the pre-preset checkpoint `30a3493` with the
coherent preset implementation in `d2c2939` on Apple M1 Max / Metal.

## Visual result

The production TAA/upscale/composite chain renders a high-frequency grid and
alternating thin geometry at legacy scale 0.50, the balanced 0.75 tier, and
native Ultra. A four-neighbor luma Laplacian measures resolved output detail;
whole-frame difference to native Ultra is the independent fidelity check.

| Configuration | Detail energy | Mean difference to native |
|---|---:|---:|
| Legacy 0.50 | 3.7882 | 2.1083 |
| Balanced 0.75 | 4.9969 | 1.3781 |
| Native Ultra | 4.7173 | 0 |

The balanced tier resolves 31.9% more detail than legacy half scale and is
34.6% closer to native. Native has slightly lower Laplacian energy because
actual subpixel resolution produces smoother anti-aliased edges rather than
the 0.75 image's residual high-frequency edge energy; the native-reference
difference prevents treating aliasing as quality.

The engine's first-run scale is now 0.75. Presets explicitly select 0.50,
0.67, 0.75, 0.85, and 1.00 from Off through Ultra. TAA, upscale filtering,
and composite sharpening are part of the same policy. TAA toggles no longer
change resolution or rebuild resolution-dependent targets.

## Matched performance

The code-path comparison holds effects and render scale at the former Ultra
configuration (preset 4, scale 0.50). Four frozen before/after binaries ran in
alternating order with 300 warm-up and 900 measured 1920x1080 frames.

| Full-frame CPU metric | Before median | After median | Change |
|---|---:|---:|---:|
| Mean | 2.2128 ms | 2.2102 ms | -0.12% |
| P50 | 2.4404 ms | 2.4242 ms | -0.66% |
| P95 | 2.7514 ms | 2.6949 ms | -2.05% |

Traced uploads remain exactly 23,264 bytes/frame. The policy adds no image,
buffer, history, render pass, draw, or per-frame upload. Every preset keeps
the separate CAS pass disabled and uses sharpening already present in the
final composite, avoiding both a second pass and double halos.

## Tier budgets

Each tier ran the same controlled scene for 300 warm-up plus 900 measured
frames at 1920x1080. These rows intentionally contain different workloads:
they are budgets for the advertised quality choices, not before/after
regression comparisons.

| Preset | Scale / extent | Graph passes | Mean | P50 | P95 |
|---|---|---:|---:|---:|---:|
| Off | 0.50 / 960x540 | 11 | 0.6683 ms | 0.2694 ms | 1.8148 ms |
| Low | 0.67 / 1286x724 | 12 | 0.9430 ms | 0.5533 ms | 2.2064 ms |
| Medium | 0.75 / 1440x810 | 15 | 2.4284 ms | 2.5579 ms | 4.2685 ms |
| High | 0.85 / 1632x918 | 24 | 3.2601 ms | 3.0191 ms | 4.6792 ms |
| Ultra | 1.00 / 1920x1080 | 24 | 4.4094 ms | 4.4255 ms | 5.8713 ms |

All tiers stay below the 16.67 ms 60-fps budget in this controlled workload;
Off and Low preserve explicit headroom for slow GPUs. Existing
resolution-dependent targets scale with the chosen render-pixel count; no
new target topology is introduced. Renderer-owned frame CPU capacity is the
same 1,918,560 bytes in every tier.

## Regression qualification

The full quick lane passed on the qualified tree:

- 325 unit tests passed, 1 hardware/file-watcher test ignored;
- 57 runnable GPU goldens passed, 2 long-running diagnostic goldens ignored;
- all 4 render-target integration tests passed;
- native FFI/schema parity, web FFI parity/arity, Wasm compilation, strict
  clippy, formatting, quality governance, visual-metric fault tests, and the
  canonical example inventory passed.

The existing half-scale TAA golden and native-resolution hardware-ray goldens
now select their intended scale explicitly. No comparison threshold or golden
reference was changed.

## Commands

```sh
cargo test --manifest-path native/shared/Cargo.toml \
  --test golden_render quality_presets -- --nocapture

cargo test --manifest-path native/shared/Cargo.toml \
  --test render_targets -- --nocapture

BLOOM_RENDER_PERF_ENGINE_REVISION=<revision> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 300 --frames 900 \
  --quality-preset 4 --render-scale 0.5 --out <matched-report.json>

BLOOM_RENDER_PERF_ENGINE_REVISION=<revision> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 300 --frames 900 \
  --quality-preset <0..4> --out <tier-report.json>

./scripts/ci-check.sh --quick \
  --summary target/ci/quick-quality-presets.json
```
