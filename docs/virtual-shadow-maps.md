# Virtual shadow maps

Bloom's directional virtual shadow map (VSM) path is an opt-in, high-tier
prototype tracked by issue #132. Enable it before process startup:

```sh
BLOOM_VSM=1 ./game
```

The default path remains the established cascaded shadow map (CSM). When VSM
is not requested, Bloom allocates no VSM textures or buffers and compiles the
canonical shaders without VSM bindings or sampling branches.

## Current contract

- Three directional levels, each with a 32 by 32 virtual address space.
- 128 by 128 interior texels and a two-texel gutter on each physical page.
- One deterministic 256-page pool. Its depth storage is 17,842,176 bytes;
  current page-table, parameter, and bounded render-uniform overhead is
  2,109,648 bytes.
- Receiver-driven demand capped at 144, 64, and 16 pages by level.
- Receiver coverage uses one fixed 1,024-entry R32 counter domain per level
  plus a sparse touched-address list. Ranking performs no hash-table work and
  retains byte-for-byte compatibility with the prior deterministic oracle.
- The fixed CPU receiver oracle is the production default. An explicit
  `BLOOM_VSM_GPU_RECEIVER=1` experiment can mark continuously changing sets of
  1,024 through 4,096 camera-visible receivers in a bounded compute pass, but
  direct timing has not qualified automatic activation.
- Experimental GPU receiver resources are lazy. The first result must exactly
  match the complete ordered CPU demand before later asynchronous results are
  consumed. Failure disables the backend, and projection transitions remain
  synchronous CPU work.
- On capable native adapters, page lists with at least 48 visible rigid-opaque
  shared-geometry casters use compact multi-draw indirect submission. Exact
  page classification remains on the CPU; small and compatibility lists keep
  direct submission.
- VSM caster-indirect resources are lazy and bounded to 80 bytes of caster
  data plus 20 bytes of command data per active record.
- A default render budget of eight dirty pages per frame.
- Missing, dirty, denied, and deferred pages always sample live CSM.
- Static page depth persists until its light matrix, caster signature, or
  affected dynamic footprint changes.
- New resident static pages cross-fade from CSM over eight frames.

Each directional level now has an independent, camera-centered orthographic
projection. The camera's light-space X and Y origin is snapped to one virtual
page footprint, so sub-page camera motion leaves the matrix and cache
byte-stable. Coverage is derived from the established cascade split with a
guard for the snapped origin, filter footprint, and receiver bias. Scene depth
bounds are pancaked and quantized independently from the planar origin.

The three clipmap matrices are carried in a 208-byte sampling uniform and are
used consistently for receiver demand, physical-page rendering, and VSM
sampling. CSM retains its own fitted matrices. A receiver outside a clipmap or
on any missing page samples that original CSM projection.

Crossing a snapped planar origin now shifts the affected level's virtual
owners while preserving their physical layers, depth, age, and content
signatures. Only pages shifted outside the 32 by 32 address space are freed.
Any prior dynamic-overlay pages are made missing before the shift because
their depth is frame-specific.

Preservation is deliberately strict. The old and new projections must have
identical light basis, scale, and depth fields; only their planar page origins
may differ. A light, depth range, content signature, or unexpected matrix
change invalidates the affected level before upload. Missing and invalidated
pages continue to sample CSM.

## Dynamic and skinned casters

Bounded dynamic caster AABBs are projected into every directional level with a
two-page filter and jitter guard. Pages nearest each caster's projected core
are scheduled first, including when multiple casters are separated in light
space.

At most four affected pages and 64 total page draws are rebuilt per frame.
Each selected page is cleared and rendered with both static and current-frame
dynamic geometry. Opaque, alpha-tested, foliage-deformed, and skinned shadow
pipelines use the same geometry, cutout bindings, joint palette, and wind
parameters as live CSM.

Every affected resident page is invalidated before its page table is uploaded.
Only a successfully rendered page becomes sampleable. Guard pages or pages
rejected by either hard budget stay dirty and therefore resolve to current
CSM; stale animated depth is never exposed. Successfully rendered dynamic
pages do not cross-fade from a differently filtered CSM shadow.

Small receiver demand (fewer than 128 pages) keeps whole-frame CSM because the
VSM indirection and page rebuild cannot repay their cost. Invalid or unbounded
dynamic bounds also select whole-frame CSM. These policies are conservative
quality and performance fallbacks, not failure states.

## Telemetry

Quality telemetry reports the VSM state under
`renderer_paths.virtual_shadows`:

- `requested`, `active`, `fallback`, and `dynamic_fallback_mode`;
- physical capacity and depth/metadata/total bytes;
- receiver demand source and count;
- receiver-bounds count and active marking backend;
- GPU receiver enablement, exact-validation state, in-flight work, dispatches,
  completions, validation failures, and lazy allocation bytes;
- resident, dirty, hit, miss, eviction, denial, invalidation, and render
  counts;
- clipmap level rebases and pages preserved or dropped by those rebases;
- per-level resident and dirty counts;
- dynamic overlay footprint, rendered pages, draws, deferred pages, and both
  hard budgets.

`renderer_paths.vsm_gpu_casters` separately reports whether the native
indirect path is available and active, considered pages, the maximum per-page
candidate count, indirect pages, caster records, indirect calls,
classification source, and lazy allocation bytes.

Expected dynamic modes are:

- `none`: no bounded dynamic footprint;
- `page-overlay`: every demanded affected page fit this frame's budgets;
- `bounded-page-overlay-with-csm`: selected pages use VSM and deferred pages
  use CSM;
- `whole-frame-csm`: the receiver footprint is small or a caster is
  unbounded.

The two occupancy images emitted by a quality capture are
`virtual-shadow-pages.png` and `virtual-shadow-physical.png`.

## Reproducible dynamic qualification

`quality-motion` has an opt-in large receiver. Its ordinary quality fixture is
unchanged unless `--vsm-dynamic` is present:

```sh
cd examples/quality-motion
perry compile main.ts
BLOOM_VSM=1 ./main \
  --vsm-dynamic \
  --quality-preset 3 \
  --render-scale 1 \
  --quality-run 120 120 0.016666666667 \
  /tmp/vsm-dynamic.png \
  /tmp/vsm-dynamic.json \
  /tmp/vsm-dynamic-intermediates
```

Qualification evidence for the current dynamic overlay milestone is in
`docs/evidence/issue-132-dynamic-vsm-overlay-v1.md`.

Qualification evidence for the independent page-snapped directional clipmap
milestone is in `docs/evidence/issue-132-directional-clipmap-v1.md`.

Add `--vsm-scroll` beside `--vsm-dynamic` to alternate the camera across an
exact light-plane page boundary every 30 frames. This opt-in transition oracle
does not alter the ordinary fixture. Qualification evidence for rolling page
preservation during that camera motion is in
`docs/evidence/issue-132-clipmap-scroll-v1.md`.

Add `--vsm-light-motion` beside `--vsm-dynamic` to alternate the primary
directional-light basis every 30 frames. The capture frame returns to the
ordinary fixture direction, allowing a direct comparison between the
one-frame invalidation and a settled cache. Qualification evidence is in
`docs/evidence/issue-132-moving-light-v1.md`.

The fixed-address receiver request-compaction oracle is qualified in
`docs/evidence/issue-132-request-compaction-v1.md`. It establishes the bounded
CPU reference and storage ABI for later compute marking without introducing a
GPU readback or changing request order.

Add `--vsm-gpu-receivers` beside `--vsm-dynamic` to create an opt-in,
continuously moving set of at least 1,024 camera-visible receive-only bounds.
The tiny stress nodes remain below the ground and therefore do not appear in
the capture. GPU marking itself is experimental and default-off after direct
pass instrumentation showed that it did not meet the no-regression
performance gate on the qualification adapter. Enable it explicitly:

```sh
BLOOM_VSM=1 BLOOM_VSM_GPU_RECEIVER=1 ./main \
  --vsm-dynamic \
  --vsm-gpu-receivers \
  --quality-preset 3 \
  --render-scale 1 \
  --quality-run 120 120 0.016666666667 \
  /tmp/vsm-gpu-receiver-control.png \
  /tmp/vsm-gpu-receiver-control.json \
  /tmp/vsm-gpu-receiver-control-intermediates
```

Omit `BLOOM_VSM_GPU_RECEIVER`, or set it to `0`, for the same-revision fixed
CPU control. Exactness, fallback safety, and the rejected performance
qualification are recorded in
`docs/evidence/issue-132-async-gpu-receiver-v1.md`. The retained experimental
path reads its dense result without a same-frame wait and compacts it with the
exact CPU oracle. It must not be enabled automatically until a new direct
GPU-cost qualification proves a net win.

`quality-stress` has an independent caster-submission fixture:

```sh
perry compile examples/quality-stress/main.ts -o examples/quality-stress/main
BLOOM_VSM=1 ./examples/quality-stress/main \
  --vsm-gpu-casters \
  --quality-run 60 180 0.016666667 \
  /tmp/vsm-indirect.png \
  /tmp/vsm-indirect.json
```

Add `BLOOM_VSM_GPU_CASTERS=0` for the same-revision CPU control. The fixture
uses 512 casting/receiving nodes and alternates the directional light every
30 frames; it does not change the ordinary 10,240-node quality-stress case.
Exact image, direct-pass, end-to-end, fallback, and lazy-resource evidence is
in `docs/evidence/issue-132-vsm-caster-indirect-v1.md`.

Add `--vsm-contact-detail` to `quality-motion` for the fixed 266-post
directional contact-detail oracle. After capturing once with `BLOOM_VSM=1`
and once without it, run:

```sh
python3 tools/quality/shadow_detail.py \
  --vsm /tmp/vsm-detail.png \
  --csm /tmp/csm-detail.png \
  --output /tmp/vsm-contact-detail.json
```

The gate measures a common neutral-ground mask, excludes colored geometry,
and requires stronger high-percentile contact edges, more retained strong-edge
pixels, and more shadow contrast than CSM. Synthetic sharp/blurred and
chromatic-exclusion controls run in the quick CI lane. Qualification evidence,
including deterministic images, incremental memory, and timings, is in
`docs/evidence/issue-132-contact-detail-v1.md`.

## Work that remains on issue #132

- Compact requests and schedule page residency entirely on the GPU so the
  current bounded dense asynchronous readback can be removed.
- Add true GPU caster classification plus indirect-count compaction on
  backends where it outperforms the shipped CPU-exact compact list.
- Independently qualify indirect caster paths for cutout, skinned,
  foliage-motion, dynamic-overlay, instanced, and dedicated-buffer geometry.
- Add explicit, default-off spot and point shadow requests, their virtual
  projections, and deterministic shared-pool arbitration. Existing unshadowed
  point lights must remain behaviorally and performance compatible.
- Qualify Bistro camera/light motion, forced small pools, alpha foliage, and
  at least 100 explicitly shadow-requesting local lights. The fixed
  directional geometric-contact oracle is now qualified.
- Integrate quality tiers and enable by default only after those gates pass.
