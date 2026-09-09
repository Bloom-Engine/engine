# Issue #128 bloom pass-tail v1 evidence

This checkpoint reduces Metal bloom cost without changing the governed
nine-tap upsample filter. The exact renderer commit under test is
`5e543a0c5a8ec94105ae7e5365851b7235abf801`; its comparison base is
`f101159a1ebd4c4b8e503ab375418e0a5fa08ade`.

## Root-cause isolation

The first candidate reconstructed the nine-tap upsample with four bilinear
half-texel samples. Static image comparisons passed, but balanced exact-commit
Metal repeats showed no reliable performance gain:

| Case | Base bloom median (ms) | Four-fetch median (ms) | Change |
|---|---:|---:|---:|
| PBR spheres high | 12.107 | 12.367 | +2.2% |
| Sponza interior | 27.061 | 26.810 | -0.9% |

That candidate was rejected. It did not justify changing the shader contract.

Per-level timestamps then exposed a large cost floor even at the smallest
Sponza levels. Downsample levels 0..4 measured 5.113, 5.051, 2.670, 2.602,
and 2.484 ms; upsample levels 0..3 measured 1.695, 1.912, 2.120, and 2.319
ms. A 45x25 render target therefore cost nearly as much as materially larger
levels. This evidence indicates Metal render-pass/store-load overhead, rather
than texture fetch count, dominates the tail of this chain.

The retained candidate removes only the smallest bloom level. The chain moves
from five levels/nine render passes to four levels/seven render passes. The
original threshold downsample and nine-tap upsample remain intact. Stable
per-level profiler labels make future cost attribution explicit.

## Exact alternating Metal results

Each row is the median of three clean exact-commit runs. Runs were ordered in a
balanced candidate/base/base/candidate/candidate/base sequence. Every run used
120 warmup frames and 300 measured frames with uncapped presentation and GPU
timestamps on Apple M1 Max / Metal.

| Case | Metric | Base median (ms) | Candidate median (ms) | Change |
|---|---|---:|---:|---:|
| PBR spheres high | bloom GPU mean | 9.891 | 7.836 | -20.8% |
|  | total GPU mean | 19.191 | 16.754 | -12.7% |
|  | total GPU p95 | 23.602 | 22.181 | -6.0% |
|  | wall mean | 8.689 | 8.416 | -3.1% |
| Sponza interior | bloom GPU mean | 27.431 | 20.890 | -23.8% |
|  | total GPU mean | 46.879 | 39.165 | -16.5% |
|  | total GPU p95 | 77.349 | 58.793 | -24.0% |
|  | wall mean | 14.033 | 12.600 | -10.2% |
| Weighted transparency | bloom GPU mean | 7.249 | 7.350 | +1.4% |
|  | total GPU mean | 20.314 | 20.458 | +0.7% |
|  | total GPU p95 | 31.007 | 30.854 | -0.5% |
|  | wall mean | 22.132 | 22.188 | +0.3% |

The transparency deltas are inside the governed same-machine noise envelope;
the candidate is neutral there. No budget or noise bound was changed.

## Image equivalence

Exact base/candidate final images were compared with `bloom-diff`:

| Case | SSIM | Luma RMSE | Pixels above 0.02 tolerance | Result |
|---|---:|---:|---:|---|
| PBR spheres high | 1.000000000 | 0.000000000 | 0% | exact |
| Sponza interior | 0.999998748 | 0.000070276 | 0% | pass |
| Weighted transparency | 1.000000000 | 0.000000000 | 0% | exact |

Sponza's maximum channel error is 0.015686274, below the comparison tolerance,
and its mean OKLab and edge deltas are 0.000002034 and 0.000002416. The static
review composite shows no visible regression. Motion review and baseline
approval remain separate human gates.

## Clean nine-case candidate corpus

The complete manifest ran from the clean exact candidate commit. All nine
captures completed; high-tier cases produced 30 required intermediates and the
constrained case produced eight. Every case reported zero steady-state bind
group, graph, pipeline, transient-buffer, and transient-texture creation, with
exactly one frame-submission encoder.

Applicable hard-budget results from that full run were:

| Case | CPU p95 / budget (ms) | GPU p95 / budget (ms) | Result |
|---|---:|---:|---|
| PBR spheres high | 2.240 / 5.000 | 23.461 / 20.000 | fail GPU |
| Damaged Helmet | 2.084 / 6.000 | 12.181 / 12.000 | fail GPU by 0.181 ms |
| Sponza interior | 2.654 / 10.000 | 56.685 / 55.000 | fail GPU |
| Skinned alpha motion | 3.191 / 10.000 | 27.497 / 30.000 | pass |
| Weighted transparency | 3.211 / 12.000 | 30.841 / 30.000 | fail GPU |
| Masked alpha coverage | 3.088 / 12.000 | 20.388 / 30.000 | pass |

Bistro exterior and draw/light stress declare RTX 4080 / Vulkan budgets, so
their M1 Max measurements remain report-only. All nine approved baselines are
still intentionally absent; no review bundle has been installed without
explicit human approval.

## Validation and remaining gates

- 70 quality-governance Python tests passed;
- three `bloom-diff` tests passed;
- 27 focused renderer shader tests passed;
- three exact base and three exact candidate runs completed for each of PBR,
  Sponza, and weighted transparency;
- three exact image comparisons passed;
- the clean exact-commit nine-case capture completed with no missing required
  intermediates and no steady-state resource churn;
- formatting and Git whitespace checks passed.

This is a stable performance checkpoint, not closure of issue #128. PBR,
Sponza, and weighted transparency remain above their Metal GPU p95 budgets;
Damaged Helmet needs repeat confirmation around its 0.181 ms miss. Human
baseline approval, motion review, and hosted Vulkan qualification also remain
open.
