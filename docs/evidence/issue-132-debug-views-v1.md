# Issue #132 complete VSM debug-view qualification v1

This evidence qualifies the debug-view acceptance criterion implemented at
`840eb49fa3190183c01cae29d81945045fe69ea5` on an Apple M1 Max using
Metal and Bloom's native high-end profile. It covers virtual pages, physical
occupancy, misses, invalidations, clip levels, and per-light cost.

These are page-grid diagnostics, not geometry rendered into the game view.
Their rectangular shapes deliberately represent virtual or physical cache
cells.

## Capture contract

An opt-in intermediate capture now writes four mutually consistent artifacts:

- `virtual-shadow-pages.png` is a 32 by 96-cell virtual address-space map.
  The three 32 by 32 clip levels are stacked vertically from near to far.
- `virtual-shadow-physical.png` is the compact physical pool, in physical-slot
  order. It exposes occupancy, fragmentation, and eviction placement.
- `virtual-shadow-legend.png` is a machine-readable six-cell palette.
- `virtual-shadow-report.json` is a same-frame VSM state and cost snapshot.
  Using a capture-time sidecar avoids comparing post-measurement images with
  telemetry serialized one frame earlier.

The stable palette is:

| State | RGB | Meaning |
| --- | --- | --- |
| free | `#080808` | no virtual owner / unallocated slot |
| miss-unrendered | `#ffb423` | resident miss that has never produced valid depth |
| invalidated | `#ff37be` | previously rendered depth made dirty |
| clip-level-0 | `#46d26e` | valid near-level page |
| clip-level-1 | `#4696ff` | valid middle-level page |
| clip-level-2 | `#be64ff` | valid far-level page |

The report also publishes a directional-light cost row containing requests,
hits, misses, invalidations, renders, residency, dirty pages, rebases, dynamic
overlay draws, owned physical depth bytes, shared pool bytes, shared metadata
and staging bytes, and the page render budget.

## Automated fail-closed gate

`tools/quality/vsm_debug_views.py` decodes the PNGs without third-party
dependencies and rejects:

- an unknown or reordered color;
- partial cells or incorrect virtual/physical dimensions;
- disagreement between the virtual and physical non-free page counts;
- occupancy, dirty-state, or per-level clean counts that disagree with the
  same-frame report;
- missing explicitly required miss or invalidation states;
- changed artifact filenames or palette metadata;
- any per-light counter or byte cost inconsistent with the top-level state.

Seven unit tests include negative controls for missing required event colors,
unknown colors, a reordered legend, per-light cost disagreement, and clip
level disagreement. Rust tests independently prove that never-rendered and
previously-rendered dirty pages receive different colors, all three clean
levels retain stable colors, free slots remain distinct, and the legend is
exact.

## Live GPU captures

Three captures exercise complementary states. Pixel counts below are
normalized to page cells; the output scale is four pixels per cell.

| Capture | Resident | Free virtual | Miss | Invalidated | Level 0 | Level 1 | Level 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| early dynamic fill | 224 | 2,848 | 212 | 0 | 5 | 7 | 0 |
| frame-30 light transition | 236 | 2,836 | 126 | 102 | 4 | 4 | 0 |
| settled static contact scene | 224 | 2,848 | 0 | 0 | 144 | 64 | 16 |

Every capture had zero unknown-color pixels. Virtual and physical page counts
were identical for every non-free state. The static control exactly exposed
the expected 144/64/16 near/middle/far demand distribution with zero dirty
pages. The transition captured 102 still-dirty previously rendered pages in
magenta and 126 never-rendered pages in amber. Its event counters recorded
105 invalidations and 12 new cache misses; pages rendered during the frame
correctly left the dirty-color population.

The transition per-light row reconciled to 236 resident pages, 228 dirty
pages, eight rendered pages, 224 requests, 212 hits, 12 misses, 105
invalidations, eight dynamic overlay draws, the hard eight-page render budget,
16,448,256 bytes of depth owned by resident pages, a fixed 17,842,176-byte
physical pool, and 2,109,648 shared metadata/staging bytes.

Reproduction from `examples/quality-motion`:

```sh
export BLOOM_VSM=1

export BLOOM_QUALITY_INTERMEDIATES=/tmp/vsm-debug-miss
./main --vsm-dynamic \
  --quality-run 1 1 0.016666667 \
  /tmp/vsm-debug-miss.png /tmp/vsm-debug-miss-telemetry.json \
  /tmp/vsm-debug-miss

export BLOOM_QUALITY_INTERMEDIATES=/tmp/vsm-debug-invalidation
./main --vsm-dynamic --vsm-light-motion \
  --quality-run 1 28 0.016666667 \
  /tmp/vsm-debug-invalidation.png \
  /tmp/vsm-debug-invalidation-telemetry.json \
  /tmp/vsm-debug-invalidation

export BLOOM_QUALITY_INTERMEDIATES=/tmp/vsm-debug-static
./main --vsm-contact-detail \
  --quality-run 120 1 0.016666667 \
  /tmp/vsm-debug-static.png /tmp/vsm-debug-static-telemetry.json \
  /tmp/vsm-debug-static

cd ../..
python3 tools/quality/vsm_debug_views.py \
  --virtual /tmp/vsm-debug-invalidation/virtual-shadow-pages.png \
  --physical /tmp/vsm-debug-invalidation/virtual-shadow-physical.png \
  --legend /tmp/vsm-debug-invalidation/virtual-shadow-legend.png \
  --telemetry /tmp/vsm-debug-invalidation/virtual-shadow-report.json \
  --require-invalidation \
  --output /tmp/vsm-debug-invalidation-result.json
```

Run the validator with `--require-miss` for the early-fill capture and without
an event requirement for the static control.

## Rendering and performance isolation

This checkpoint changes no shader, render pass, texture, buffer, bind group,
draw submission, page request, cache policy, VSM sampling, or CSM fallback.
The page-state images were already generated only when an intermediate
capture was requested; the change separates their dirty-state colors and adds
a legend plus a same-frame JSON sidecar in that same post-measurement path.
Normal frames execute none of the image generation, PNG encoding, or sidecar
I/O. The expanded capability JSON is serialized only when explicitly queried.

Consequently the production frame graph and final game image are unchanged,
and the diagnostic work remains excluded from measured performance windows.
Strict Clippy correctness/suspicious/performance policy and the quality
contract passed. The complete quick lane passed with FFI parity on every
backend, 349 shared unit tests plus one ignored, negotiated headless device
construction, 59 GPU goldens plus two hardware-policy ignores, four render
target tests, Web/WASM checking, 39 quality tests, and all 20 canonical
examples.

Machine-readable evidence is in
`docs/evidence/issue-132-debug-views-v1.json`.
