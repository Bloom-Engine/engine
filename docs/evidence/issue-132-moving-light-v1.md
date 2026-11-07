# Issue #132 moving-light qualification v1

This checkpoint qualifies the conservative directional-light basis-change
path at revision `2f73135cc454fc81d8661db9b887b56dee7a0254` on an Apple
M1 Max using Metal and Bloom's native high-end profile.

## Qualified behavior

`quality-motion --vsm-dynamic --vsm-light-motion` alternates between two
directional-light vectors every 30 frames. Frame 240 returns to the ordinary
fixture direction. Its output can therefore be compared directly with a
settled capture using the same camera, geometry, animation time, and final
light while telemetry observes the transition frame.

A changed light basis cannot satisfy the clipmap scroll key. All three levels
therefore take the conservative invalidation path. Every clean resident page
becomes dirty before the GPU page table is uploaded, so it is unsampleable and
resolves through the newly rendered live CSM. Virtual ownership remains cached
to avoid allocation churn, but cached ownership never makes dirty depth
visible.

All three exact-source transition captures reported:

- zero clipmap rebases and zero preserved or dropped pages;
- 108 clean pages invalidated in the transition frame;
- eight bounded page renders and eight pending page requests;
- 236 resident pages, of which 228 remained dirty and used CSM fallback;
- 224 requested pages and 224 cache hits;
- zero misses, allocation denials, or evictions;
- 153 guarded dynamic pages, four rendered dynamic pages, and eight overlay
  draws;
- the unchanged fixed VSM allocation of 19,951,824 bytes.

The cache-hit count represents retained virtual-to-physical ownership. The
dirty count and missing page-table entries are the sampling-safety authority.

## Image evidence

The three one-frame transition captures were effectively identical:

- luminance RMSE at most `0.000023677`;
- luminance SSIM at least `0.999999821`;
- `0.0%` of pixels above the 0.02 tolerance;
- mean OKLab delta at most `0.000000101`;
- mean edge delta at most `0.000000133`.

Against the settled cache at the same final light, the transition frame passed
the repository image gate with RMSE `0.006145390` and SSIM `0.999100208`.
The intended CSM/VSM fallback and temporal settling affected `2.612083435%`
of pixels above the local 0.02 threshold, with mean edge delta `0.000771535`.

Against a CSM-only run with the identical moving-light history, the transition
frame also passed with RMSE `0.008730293` and SSIM `0.992735803`. The
differences are localized to the expected foliage and shadow detail rather
than virtual-page boundaries.

Visual review of the full images, composites, and heatmaps found no page
rectangles, missing shadow bands, newly unshadowed regions, or persistent
old-direction shadows.

## Bounded work and compatibility

The median across three continuously moving-light runs was:

- wall time `14.385408 ms`;
- shadow CPU `0.218197 ms`;
- shadow GPU `2.969117 ms`;
- virtual-page CPU `0.044228 ms`;
- virtual-page GPU `2.541226 ms`.

This oracle deliberately forces a full directional invalidation every 30
frames. Work remains hard-capped at eight page renders per frame, and no
allocation growth or churn occurred. The flag is opt-in test behavior; no
renderer code or ordinary example path changed in this checkpoint.

The preceding full quick lane already qualified the exact renderer source:
FFI parity including Linux/Web, strict Clippy, 340 shared tests with 1 ignored,
headless device construction, 59 GPU goldens with 2 policy ignores, 4
render-target tests, WASM, quality/cooker tooling, and 20 canonical examples.
The committed TypeScript oracle also compiled successfully through Perry.

Machine-readable measurements accompany this note in
`docs/evidence/issue-132-moving-light-v1.json`.

Directional camera and light transition foundations are now qualified.
GPU-driven receiver marking/request compaction/caster submission, spot and
point projections, stress-scene coverage, and tier rollout remain open.
