# Issue #132 directional clipmap qualification v1

This checkpoint qualifies independent, page-snapped directional virtual-shadow
clipmaps at revision `38615844851e4cf09fe499e161e8b6863f6725b7` on an Apple
M1 Max using Metal and Bloom's native high-end profile.

## Qualified behavior

The prior VSM path rendered and sampled virtual pages through the fitted CSM
matrices. The new path computes three camera-centered orthographic projections
from the established cascade distances. Each light-space planar origin snaps
to one virtual-page footprint; scene depth is pancaked and quantized
separately. Sub-page motion therefore leaves every matrix byte-stable.

Receiver marking, physical-page rendering, scene shading, and material shading
all use the same clipmap matrices. The independently fitted CSM matrices remain
live fallback data. Missing, dirty, deferred, and out-of-volume VSM samples
continue through CSM instead of exposing stale or unshadowed pixels.

At the final dynamic-fixture frame the exact revision reported:

- projection `camera-centered-page-snapped-clipmap`;
- 224 receiver-demand pages and 224 resident pages;
- 153 conservatively guarded dynamic pages;
- 4 rendered dynamic overlay pages and 8 total page draws;
- 119 demanded pages left dirty for current CSM fallback;
- 0 denied pages and 0 evictions;
- a fixed 19,951,824-byte VSM allocation, 192 bytes above the prior milestone
  for the three sampling matrices.

The static Sponza fixture reached 224 resident pages with zero dirty pages,
denials, evictions, or steady-state page renders.

## Image evidence

Three exact-revision dynamic captures used 120 warmup and 120 measured frames.
The widest repeat delta was:

- luminance RMSE `0.000021062`;
- luminance SSIM `0.999999940`;
- `0.0%` of pixels above the 0.02 tolerance;
- mean OKLab delta `0.000000213`;
- mean edge delta `0.000000215`.

Against the fitted-matrix dynamic control, the new projection changed the
intended foliage and ground-shadow detail while remaining inside the
repository's image gate:

- luminance RMSE `0.006658406`;
- luminance SSIM `0.994245768`;
- mean OKLab delta `0.001265500`;
- mean edge delta `0.001250187`.

The exact-revision static Sponza comparison likewise passed with RMSE
`0.005528215`, SSIM `0.996063292`, mean OKLab delta `0.002201730`, and mean
edge delta `0.001407787`. Visual review of both full composites and heatmaps
showed localized shadow-edge and foliage-detail changes, with no page
rectangles, missing shadow regions, or newly unshadowed geometry.

With `BLOOM_VSM` unset, the prior and exact-revision captures remained
effectively identical: RMSE `0.000013567`, SSIM `1.000000000`, and zero pixels
above tolerance. Telemetry reported zero VSM capacity, bytes, residency, and
work.

## Bounded performance evidence

The preceding fitted-overlay qualification used four measurements on the same
machine and fixture. The exact clipmap revision used three measurements with
the same 120 warmup, 120 measured frames, fixed 60 Hz timestep, quality preset
3, and native render scale.

All three median steady-state values moved down:

- wall time: `13.760610` to `13.025648 ms`;
- shadow CPU: `0.180561` to `0.172756 ms`;
- shadow GPU: `4.100516` to `4.029285 ms`.

The exact-revision median bounded page work was `0.031248 ms` CPU and
`3.696149 ms` GPU. This remains opt-in behind `BLOOM_VSM=1`; the default path
does not compute clipmaps, allocate VSM resources, add render passes, or inject
VSM shader code.

## Regression gates

- FFI/schema parity passed for every declared platform.
- Formatting, strict correctness/performance Clippy, and the file-size ratchet
  passed.
- Native release compilation and the `wasm32-unknown-unknown` Web-feature
  check passed.
- 336 shared unit tests passed with 1 ignored.
- The headless negotiated-device test passed.
- 59 GPU goldens passed with 2 hardware-policy tests ignored.
- All 4 render-target tests passed.
- 29 focused VSM tests passed, including uniform layout, WGSL injection,
  page-stable sub-page motion, snapped rebase, scene-depth containment,
  guarded invalidation, and fixed memory bounds.

Machine-readable measurements accompany this note in
`docs/evidence/issue-132-directional-clipmap-v1.json`.
