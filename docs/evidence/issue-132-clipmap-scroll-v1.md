# Issue #132 clipmap-scroll qualification v1

This checkpoint qualifies physical-page preservation across directional
clipmap camera rebases at revision
`f4da02121646799d156133236ebe1c252e111933` on an Apple M1 Max using
Metal and Bloom's native high-end profile.

## Qualified behavior

Each level now retains an integer planar page origin and an exact cache key for
the projection fields that must remain stable. When only that origin changes,
resident virtual owners shift to their new coordinates while keeping the same
physical layers, depth, age, and content signatures. Owners shifted beyond the
32 by 32 virtual extent are the only pages freed.

The scroll is not used when the light basis, scale, depth projection, or
content signature changes. Those cases invalidate the affected level before
the page table is uploaded. Old dynamic-overlay pages are likewise invalidated
before any remap, so current-frame animation can never sample preserved stale
depth. Missing, dirty, dropped, or invalidated pages continue through live CSM.

An opt-in `--vsm-scroll` oracle moves the camera and target by an exact vector
in the qualification sun's light plane every 30 frames. Frame 240 crosses
back over the snapped boundary, so the final image and telemetry observe the
rebase rather than a settled frame.

All three exact-revision transition captures reported:

- one clipmap-level rebase and 152 preserved physical pages;
- zero pages dropped from the occupied footprint;
- 224 requested pages, 224 hits, and zero misses;
- zero evictions and zero denied allocations;
- 232 resident pages and 125 dirty pages;
- 13 safe invalidations, eight rendered pages, and eight pending pages;
- 153 guarded dynamic pages, four rendered overlay pages, and eight overlay
  draws;
- the unchanged fixed VSM allocation of 19,951,824 bytes.

The cache unit oracle separately shifts owners across a virtual boundary and
proves that overlapping physical pages remain resident while only the outgoing
edge is dropped.

## Image evidence

The three moving-camera captures were effectively identical:

- luminance RMSE at most `0.000009965`;
- luminance SSIM `1.000000000`;
- `0.0%` of pixels above the 0.02 tolerance;
- mean OKLab delta at most `0.000000041`;
- mean edge delta at most `0.000000045`.

Visual review of the transition frame found no page rectangles, missing shadow
bands, newly unshadowed regions, or discontinuous foliage shadows.

The stable-camera image remained effectively identical to the preceding
page-snapped clipmap checkpoint: RMSE `0.000009865`, SSIM `1.000000000`, zero
pixels above tolerance, mean OKLab delta `0.000000037`, and mean edge delta
`0.000000051`.

With `BLOOM_VSM` unset, the ordinary fixture remained effectively identical:
RMSE `0.000005068`, SSIM `1.000000000`, and zero pixels above tolerance.
Telemetry reported zero VSM capacity, bytes, residency, or work.

## Bounded performance evidence

The implementation initially exposed transition classification on stable
frames. Qualification caught that cost and the final revision gates the remap
logic behind an actual projection or content change.

Five final stationary measurements used 120 warmup and 120 measured frames, a
fixed 60 Hz timestep, quality preset 3, and native render scale. Their medians
versus the preceding directional-clipmap checkpoint were:

- wall time: `13.025648` to `12.766289 ms`;
- shadow CPU: `0.172756` to `0.171354 ms`;
- shadow GPU: `4.029285` to `3.899865 ms`;
- virtual-page CPU: `0.031248` to `0.032132 ms`;
- virtual-page GPU: `3.696149` to `3.605031 ms`.

The sub-microsecond page-CPU delta is measurement noise within a lower total
shadow CPU result. No stable-frame cache walk, allocation, render pass, draw,
or upload was added. The path remains opt-in behind `BLOOM_VSM=1`.

## Regression gates

- FFI/schema parity passed for every declared platform.
- Formatting, strict correctness/performance Clippy, and the file-size ratchet
  passed.
- Native release compilation and the `wasm32-unknown-unknown` Web-feature
  check passed.
- 340 shared unit tests passed with 1 ignored.
- The headless negotiated-device test passed.
- 59 GPU goldens passed with 2 hardware-policy tests ignored.
- All 4 render-target tests passed.
- 33 focused VSM tests passed.
- All 20 canonical examples passed their inventory gate.

Machine-readable measurements accompany this note in
`docs/evidence/issue-132-clipmap-scroll-v1.json`.

Moving-light transitions, GPU-driven request/submission work, local-light
virtual projections, stress-scene qualification, and quality-tier rollout
remain open on issue #132.
