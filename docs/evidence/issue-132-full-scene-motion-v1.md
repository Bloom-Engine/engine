# Issue #132 full-scene VSM motion qualification v1

This evidence qualifies the fixed Sponza and Bistro camera-path acceptance
criterion at `aa36327aaee694af17a1c71701bd93c70f814c90` on an Apple M1 Max
using Metal and Bloom's native high-end profile.

Sponza is the complete checked-in Khronos asset. Bistro is Bloom's governed
camera-visible 96 unique-mesh subset derived from the pinned source revision;
this evidence does not claim full 2,909-instance Bistro streaming coverage.

## Fixed paths and controls

Both established examples now accept the opt-in `--vsm-motion-path` flag:

- Sponza moves 2.5 metres along the exact sun light-plane right vector;
- Bistro moves 6 metres along its exact sun light-plane right vector;
- each path alternates every 30 frames and returns to its ordinary camera on
  frame 240;
- the post-measurement capture therefore has the same camera geometry as its
  settled control while telemetry records the preceding clipmap rebase.

Each scene uses four captures at 1600 by 900 physical pixels: settled VSM,
settled CSM, motion VSM, and motion CSM. The oracle evaluates:

`(motion VSM - motion CSM) - (settled VSM - settled CSM)`.

This matched-control residual removes auto-exposure, TAA, SSGI, SSR, and other
legitimate path history shared by both shadow backends. What remains is the
VSM-specific transition change relative to its ordinary settled difference
from CSM.

Sponza reproduction:

```sh
cd examples/sponza

BLOOM_VSM=1 ./main --vsm-motion-path --yaw 0 --taa 1 \
  --quality-preset 3 --render-scale 1 \
  --quality-run 120 120 0.016666667 \
  /tmp/sponza-motion-vsm.png /tmp/sponza-motion-vsm.json

./main --vsm-motion-path --yaw 0 --taa 1 \
  --quality-preset 3 --render-scale 1 \
  --quality-run 120 120 0.016666667 \
  /tmp/sponza-motion-csm.png /tmp/sponza-motion-csm.json
```

Run the same commands without `--vsm-motion-path` for settled controls, then
invoke `tools/quality/vsm_motion_corpus.py` with all four images and telemetry
files. Bistro uses the corresponding `examples/bistro` commands plus the
governed `bistro-quality.gltf` prepared by `tools/quality/prepare_bistro.py`.

## Automated artifact gate

The dependency-free gate requires active VSM only in the VSM captures, an
observed clipmap rebase with preserved pages, zero denied or evicted pages,
fixed allocation, and page work within the hard render budget.

Its image thresholds reject:

- page seams and clipmap rings through coherent row, column, connected-span,
  and line-like-component tests;
- missing-page flashes through positive residual coverage;
- stale or doubled shadows through negative residual coverage;
- broad or high-contrast transition changes through RMSE, p99, changed-area,
  and largest-component limits.

Six synthetic negative controls prove that a vertical page seam, an
elliptical clipmap ring, a large unshadowed flash, a stale/doubled shadow, a
missing rebase, and malformed telemetry fail closed. A seventh test proves
the matched control removes an unrelated temporal/exposure change.

## Real image results

| Backend-isolated residual | Sponza | Bistro |
| --- | ---: | ---: |
| RMSE | 0.002120 | 0.004746 |
| Mean absolute | 0.001029 | 0.000762 |
| Absolute p95 | 0.003638 | 0.003355 |
| Absolute p99 | 0.008394 | 0.018491 |
| Pixels above 0.03 | 0.0052% | 0.4667% |
| Bright flash pixels above 0.05 | 0.0002% | 0.0944% |
| Dark/stale pixels above 0.05 | 0.0006% | 0.0642% |
| Maximum row coverage | 1.875% | 6.000% |
| Maximum column coverage | 0.444% | 11.222% |
| Largest connected component | 20 pixels | 225 pixels |
| Largest component frame ratio | 0.0014% | 0.0156% |
| Long seam/ring-like component | no | no |

Both ordinary motion VSM captures also remained inside Bloom's image gate
against their same-path CSM controls:

| Metric | Sponza | Bistro |
| --- | ---: | ---: |
| Luminance RMSE | 0.012137 | 0.006207 |
| Luminance SSIM | 0.990045 | 0.994232 |
| Mean OKLab delta | 0.003250 | 0.000980 |
| Mean edge delta | 0.001176 | 0.001186 |

Visual review of the final composites and heatmaps found no page rectangles,
long seams, clip-level rings, unshadowed flashes, or retained shadows from the
other camera position.

## Cache safety and bounded work

At the final Sponza transition, telemetry reported two level rebases, 228
preserved pages, zero dropped pages, all 224 demands as cache hits, zero
dirty pages, zero renders, zero pending pages, and zero denials or evictions.

The larger Bistro move reported three level rebases, 202 preserved and 54
safely dropped pages, 190 hits plus 34 misses, eight bounded renders, 26 dirty
pages excluded from sampling, eight pending pages against the hard
eight-page budget, and zero denials or evictions. Missing/dirty pages used
live current-camera CSM fallback. Both cases retained the fixed
19,951,824-byte VSM allocation.

Independent repeats preserved every cache/work counter exactly. Repeat image
RMSE was 0.000208 for Sponza and 0.000468 for Bistro, with SSIM 0.999997 and
0.999954 respectively.

## Regression and compatibility

This checkpoint changes no renderer, pass, shader, resource, binding, draw,
page policy, or production API. It adds opt-in example camera paths and an
offline oracle. Fixture frame bookkeeping and path math execute only when
`--vsm-motion-path` is present.

The complete `scripts/ci-check.sh --quick` lane passed at the implementation
revision: FFI/schema parity, strict Clippy, formatting, file-size ratchet,
348 shared tests plus one ignored, negotiated headless construction, shared
Web/WASM, 59 GPU goldens plus two hardware-policy ignores, four render-target
tests, 32 quality tests, cooker/diff tools, and all 20 examples.

Machine-readable evidence is in
`docs/evidence/issue-132-full-scene-motion-v1.json`.
