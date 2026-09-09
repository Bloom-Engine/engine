# Issue #132 dynamic VSM overlay qualification v1

This checkpoint qualifies bounded current-frame virtual-shadow pages for
dynamic and skinned directional casters at revision
`6b4c256bcf055204cee1af4f627274729077f43d` on an Apple M1 Max using Metal
and Bloom's native high-end profile.

## Qualified behavior

The prior revision, `dcdb29559eef4e34db9cab80569077483c01c1bc`,
masked every page touched by a dynamic caster and sampled live CSM there. The
new path retains that safety fallback but rebuilds up to four core pages with
current static and dynamic geometry. The draw budget is 64 across all selected
pages.

The qualification fixture uses the existing animated, skinned Fox and
alpha-tested Sponza curtain. `--vsm-dynamic` adds a large receive-only ground
mesh so demand is production-sized without changing the ordinary fixture.

At the final frame the new path reported:

- 224 receiver-demand pages and 224 resident pages;
- 151 conservatively guarded dynamic pages;
- 4 rendered dynamic overlay pages and 8 total page draws;
- 75 demanded guard pages deliberately left dirty for live CSM fallback;
- 0 denied pages, 0 evictions, and the fixed 19,951,632-byte total VSM budget.

The small ordinary fixture still selects `whole-frame-csm` with 67 demand
pages, no VSM residency, and no overlay work.

## Image evidence

Two repeated overlay captures were stable to one 8-bit code value:

- luminance RMSE `0.000005030`;
- luminance SSIM `1.000000000`;
- `0.0%` of pixels above the 0.02 tolerance;
- mean OKLab delta `0.000000014`;
- mean edge delta `0.000000020`.

Compared with the prior per-page CSM control, the overlay produced the intended
VSM-filtered shadow detail in the affected pages:

- luminance RMSE `0.008894581`;
- luminance SSIM `0.992135584`;
- mean OKLab delta `0.001824761`;
- mean edge delta `0.001542727`.

Visual review of the full image and diff showed changes localized to shadowed
foliage, the skinned caster, and ground shadow edges, without page rectangles,
missing shadow regions, or an unshadowed fallback.

With `BLOOM_VSM` unset, old and new ordinary captures were effectively
identical: SSIM `1.000000000`, RMSE `0.000012179`, and zero pixels above the
0.02 tolerance. Telemetry reported zero VSM capacity, bytes, residency, and
work.

## Bounded performance evidence

Three old-control and four overlay measurements used 120 warmup plus 120
measured frames at fixed 60 Hz. The median whole-frame wall time was
`13.873404 ms` before and `13.760610 ms` after. The median shadow GPU sample
was `4.008879 ms` before and `4.100516 ms` after, an absolute `0.091637 ms`
difference within the observed run-to-run range. The bounded page work added
`0.125685 ms` to median shadow CPU time (`0.054876` to `0.180561 ms`).

This quality work is confined to explicit `BLOOM_VSM=1` use. Four page passes
and 64 draws are hard ceilings; deferred work uses the already-rendered CSM.
The default renderer retains its zero-allocation, zero-pass VSM path.

## Regression gates

- FFI/schema parity passed for every declared platform.
- Formatting, strict correctness/performance Clippy, and the file-size ratchet
  passed.
- Native release and the proper `wasm32-unknown-unknown` Web-feature check
  passed.
- 331 shared unit tests passed with 1 ignored.
- The headless negotiated-device test passed.
- 59 GPU goldens passed with 2 hardware-policy tests ignored.
- All 4 render-target tests passed.
- 24 focused VSM tests passed, including guarded invalidation, missing-page
  safety, projected-core priority, separated-caster priority, memory bounds,
  deterministic demand, and shader parsing.

Machine-readable measurements accompany this note in
`docs/evidence/issue-132-dynamic-vsm-overlay-v1.json`.
