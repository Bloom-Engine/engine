# Issue #128 SSGI resolve compute v1 evidence

Renderer revision `09ad0b755af9f10083712327d7f0edb1d88f228b` removes a
Metal render-pass tail from the half-resolution SSGI resolve. Its clean exact
comparison is `5e543a0c5a8ec94105ae7e5365851b7235abf801`, the prior
four-level bloom checkpoint.

## Change and root cause

The resolve shader has no derivatives or blending, and every output pixel is
fully overwritten. It therefore does not need a fullscreen raster pass. The
new path dispatches an 8x8 compute grid, reconstructs the same pixel-center UV,
and writes the existing `rgba16float` ping-pong target through a storage
binding. The disabled-path render clear and the existing sampled/history uses
remain available.

This targets the cost exposed by the prior bloom investigation: on Metal,
small render passes retain a multi-millisecond floor that is not explained by
shader work. The resolve shader itself was not simplified.

Before the checkpoint was committed, three comparison and three candidate
runs were ordered candidate/base/base/candidate/candidate/base. The candidate
tree contained only the five files committed unchanged as `09ad0b7`.

| Case | Metric | Base median (ms) | Candidate median (ms) | Change |
|---|---|---:|---:|---:|
| PBR spheres high | resolve GPU mean | 2.798 | 0.052 | -98.2% |
|  | total GPU mean | 20.776 | 14.438 | -30.5% |
|  | total GPU p95 | 23.374 | 15.921 | -31.9% |
|  | wall mean | 6.265 | 5.877 | -6.2% |
| Sponza interior | resolve GPU mean | 5.068 | 0.056 | -98.9% |
|  | total GPU mean | 41.959 | 29.890 | -28.8% |
|  | total GPU p95 | 59.994 | 41.746 | -30.4% |
|  | wall mean | 9.938 | 9.934 | -0.04% |

## Clean exact-commit Metal corpus

The complete nine-case corpus ran from clean revision `09ad0b7` on Apple M1
Max / Metal. The result artifact SHA-256 is
`49de2b4a46d2f4ad309908777769c0de81d182a9f403afb294de59224cf24d37`.

| Case | Base GPU p95 (ms) | Candidate GPU p95 (ms) | Change | Resolve mean (ms) |
|---|---:|---:|---:|---:|
| PBR spheres high | 23.461 | 15.408 | -34.3% | 0.051 |
| PBR constrained | 5.791 | 5.965 | +3.0% | disabled |
| Damaged Helmet | 12.181 | 8.425 | -30.8% | 0.051 |
| Sponza interior | 56.685 | 41.824 | -26.2% | 0.052 |
| Bistro exterior | 44.033 | 34.862 | -20.8% | 0.031 |
| Skinned alpha motion | 27.497 | 19.830 | -27.9% | 0.045 |
| Draw/light stress | 35.643 | 26.961 | -24.4% | 0.068 |
| Weighted transparency | 30.841 | 23.648 | -23.3% | 0.052 |
| Masked alpha coverage | 20.388 | 14.886 | -27.0% | 0.072 |

The constrained case does not enable this SSGI path; its 3.0% delta is inside
the existing same-machine noise allowance. Every applicable Metal GPU p95
budget now clears: PBR 15.408/20, Helmet 8.425/12, Sponza 41.824/55, skinned
19.830/30, weighted 23.648/30, and masked 14.886/30 ms. Bistro and draw/light
stress retain their Vulkan-only hard budgets.

The long corpus run coincided with host CPU spikes in PBR, weighted, and
masked. Three clean exact-commit repeats per affected case produced median CPU
p95 values of 2.605, 7.159, and 5.240 ms respectively, all below their 5, 12,
and 12 ms limits. Their repeated GPU p95 medians were 15.877, 24.575, and
15.457 ms.

All eight high-tier cases emitted 30 required intermediates and the
constrained case emitted eight. Every case reported zero steady-state bind
group, graph, pipeline, transient-texture, and transient-buffer creation, with
exactly one frame-submission encoder.

## Image equivalence

All nine final images pass `bloom-diff` against the clean comparison corpus.
The worst final SSIM is 0.999997616 and the worst luminance RMSE is
0.000093042. Eight direct `ssgi.png` intermediate comparisons also pass; their
worst SSIM is 0.999997973, worst luminance RMSE is 0.000042475, maximum channel
error is one 8-bit level, and zero pixels exceed the configured tolerance.

This confirms that the gain comes from removing attachment/pass overhead, not
from weakening the resolve filter or its temporal ownership.

## Regression gates and remaining work

- release shared library: 483 passed, zero failed, one intentional ignore;
- release GPU golden corpus: 86 passed, zero failed, four hardware/long-run
  ignores;
- focused SSGI shader-contract tests: 9 passed;
- clean nine-case Metal capture: completed with all required artifacts;
- final-image comparisons: 9/9 passed;
- direct SSGI-intermediate comparisons: 8/8 passed;
- formatting and Git whitespace checks: passed.

This is a stable Metal performance checkpoint. The corpus still has no
installed human-approved baselines, so report-only runs continue to record
baseline-missing failures. Hosted RTX 4080 / Vulkan qualification and explicit
baseline review remain open.
