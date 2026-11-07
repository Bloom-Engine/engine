# Issue #132 bounded VSM caster indirect submission v1

This evidence qualifies revision
`7962bee2f8b64c21a2a0b1bec49369aaaf1aa678` on an Apple M1 Max using
Metal and Bloom's native high-end profile.

## Shipped boundary

Large rigid-opaque caster lists for a directional virtual-shadow page can now
use compact multi-draw indirect submission. The implementation deliberately
keeps the exact CPU page-frustum classification:

- A page needs at least 48 visible eligible draws. Smaller pages retain the
  established compatibility loop.
- Only static, rigid, opaque geometry already resident in the shared geometry
  arena is eligible.
- Cutout, skinned, foliage-deformed, dynamic-overlay, dedicated-buffer,
  overflow, unsupported, Web, and GPU-driven-disabled work remains on the
  compatibility renderer.
- CPU classification visits each caster once. It partitions the exact visible
  set without repeating the page-frustum AABB test.
- The CPU precomputes one local-to-shadow clip matrix per eligible caster and
  uploads compact 80-byte caster records plus 20-byte indexed-indirect
  commands.
- Each qualifying page submits its contiguous command range with one
  `multi_draw_indexed_indirect` call. There are no fixed Cartesian
  page-by-caster commands and no zero-instance safety draws.

This boundary is important on Metal. Bloom's negotiated WebGPU feature set
does not expose indirect-count draws there. A compute classifier would still
require fixed-count consumption, which was slower in direct experiments. Full
GPU classification/count compaction therefore remains future work rather than
being implied by this milestone.

Activation is automatic only when VSM is requested, the established
GPU-driven/shared-geometry path is available, device storage limits cover the
hard maximum, the page is not a dynamic overlay, and the 48-draw workload
threshold is met. `BLOOM_VSM_GPU_CASTERS=0` provides a same-revision control.

Resources remain lazy. The qualified 512-caster fixture rounded the active
capacity to 512 records:

- 40,960 bytes of clip-matrix/caster records;
- 10,240 bytes of indirect commands;
- 51,200 total added GPU bytes.

Small VSM work, VSM-disabled startup, GPU-driven-disabled startup, and the
explicit control all reported zero bytes and zero indirect calls.

## Reproducible fixture

`quality-stress --vsm-gpu-casters` changes only the opt-in qualification
fixture. It creates 512 shadow-casting and receiving nodes, remains below the
1,024-node page compatibility cap, and alternates the directional-light basis
every 30 frames so a 180-frame measurement contains repeated page updates.
The ordinary 10,240-node quality-stress fixture is unchanged.

The candidate command was:

```sh
BLOOM_VSM=1 ./examples/quality-stress/main \
  --vsm-gpu-casters \
  --quality-run 60 180 0.016666667 \
  /tmp/vsm-indirect.png \
  /tmp/vsm-indirect.json
```

The control added `BLOOM_VSM_GPU_CASTERS=0`. Five runs per mode were
interleaved. At capture, telemetry reported eight considered pages, a maximum
of 85 eligible casters on one page, four indirect pages, 302 caster records,
four indirect calls, and 51,200 bytes. The classification source was
`cpu-exact-prefilter+gpu-indirect-submit`.

## Exact image evidence

Candidate and same-revision CPU-control captures were byte-identical across
all 3,686,400 pixels:

- identical SHA-256:
  `06c69658f481bc335ac6dc3333cc7758d315182bb7e9b4797ee18882f28c9c01`;
- luminance and RGB RMSE `0`;
- maximum absolute error `0`;
- SSIM `1`;
- zero pixels above the 0.02 tolerance;
- mean OKLab and edge delta `0`.

The exact match covers the CPU-to-indirect matrix composition, base-vertex
bit preservation, first-instance record indexing, shared vertex/index arena,
and mixed indirect/compatibility page rendering.

## Performance qualification

Median-of-five results over 180 measured frames were:

| Domain | CPU control | Indirect candidate | Delta |
| --- | ---: | ---: | ---: |
| Wall frame mean | 11.893637 ms | 11.457747 ms | -3.66% |
| CPU frame mean | 9.824165 ms | 8.851101 ms | -9.90% |
| GPU frame mean | 20.292364 ms | 19.262754 ms | -5.07% |
| GPU frame p50 | 21.095042 ms | 19.216956 ms | -8.90% |
| VSM page CPU | 0.106694 ms | 0.094448 ms | -11.48% |
| VSM page GPU | 1.092336 ms | 0.582765 ms | -46.65% |
| Render-total CPU | 4.921546 ms | 4.435739 ms | -9.87% |

CPU frame p50 moved from 4.548665 to 4.618707 ms (+0.070042 ms) while
CPU-frame mean, render-total CPU, wall time, and the directly targeted
VSM-page CPU work all improved. The five individual CPU-frame samples had
substantially wider unrelated dispersion than that delta, so it is recorded
as noise rather than used to contradict the end-to-end and direct-pass
signals.

Driver queue-submit attribution increased by 0.057261 ms, while render-total
CPU fell by 0.485807 ms and complete CPU-frame mean fell by 0.973064 ms. This
is an internal movement of driver work, not an end-to-end CPU regression.

Two broader prototypes were rejected before commit:

- a page-by-caster Cartesian indirect table made Metal consume thousands of
  zero-instance commands;
- a GPU safety-cull still consumed a fixed compact count, added a compute
  dispatch, and shifted avoidable work into queue submission.

The shipped version removes both costs and retains only the compact,
CPU-exact command list.

## Regression gates

- The complete `scripts/ci-check.sh --quick` lane passed.
- FFI/schema parity passed for macOS, Linux, Windows, Android, iOS, tvOS,
  watchOS, and Web.
- Formatting, strict correctness/performance Clippy, and the file-size ratchet
  passed.
- 347 shared unit tests passed with 1 ignored.
- The headless negotiated-device test passed.
- 59 GPU goldens passed with 2 hardware-policy tests ignored.
- All 4 render-target tests passed.
- The `wasm32-unknown-unknown` Web-feature check passed.
- Quality governance, visual-diff, asset-cooker, and all 20 canonical-example
  inventory gates passed.

Machine-readable measurements accompany this note in
`docs/evidence/issue-132-vsm-caster-indirect-v1.json`.

Remaining caster work includes true GPU classification plus indirect-count
compaction where supported, and separately qualified cutout, skinned,
foliage-motion, dynamic-overlay, instanced, and dedicated-buffer paths.
