# Issue #132 asynchronous GPU receiver-marking experiment v1

This evidence validates the exactness and fallback behavior introduced at
`ad6e7c625c4350589d0dc5346845505ac1e6b31b` on an Apple M1 Max using
Metal and Bloom's native high-end profile. Its original automatic performance
qualification is superseded: direct pass instrumentation subsequently showed
that GPU marking cost more GPU time than the fixed CPU work it removed.
Current revisions therefore keep this backend explicit opt-in and default to
the fixed CPU oracle. That corrected default begins at
`9fe0371a8c020e56289cff5555f9110f048677d0`.

## Validated behavior

Large directional-receiver sets can now mark the fixed 32 by 32 by 3
coverage domain in a compute pass. The runtime remains deliberately
conservative:

- With `BLOOM_VSM_GPU_RECEIVER=1`, GPU marking activates only for 1,024 through
  4,096 camera-visible receiver bounds on a capable native adapter.
- Small scenes continue to use the qualified fixed-address CPU oracle. They
  create no marker pipeline, buffer, pass, copy, or readback.
- The first GPU result is compared with the complete ordered CPU demand
  vector. Only an exact match validates the backend. A mapping error or
  mismatch permanently disables it for that renderer and keeps the CPU
  result.
- Two 12,288-byte readback buffers permit asynchronous progress. Production
  code uses `device.poll(Poll)` and never waits for the result in the frame
  that produced it.
- Projection changes still compute current demand synchronously on the CPU.
  This avoids carrying receiver demand across a camera or light transition.
- Continuous receiver motion consumes the newest completed same-projection
  result and records the current bounds. The result can lag by one frame.
  Newly touched addresses absent from that result sample current CSM; existing
  resident addresses contain projection- and content-matched depth.

The compute output is the same dense R32 coverage ABI introduced by the prior
request-compaction milestone. Ranking, caps, interleaving, and page-cache
ownership still run on the CPU after asynchronous readback. Keeping that
boundary explicit avoids claiming a fully GPU-resident page scheduler before
one exists.

Resources are allocated only on the first qualifying dispatch. At 1,140
receivers the bounds buffer rounds to 2,048 entries and the complete added
allocation is 102,896 bytes:

- 65,536 bytes for receiver AABBs;
- 496 bytes for three matrices, three six-plane frusta, and counts;
- 12,288 bytes for dense atomic coverage;
- two 12,288-byte readback buffers.

Omitting `BLOOM_VSM_GPU_RECEIVER`, or setting it to `0`, provides the
same-revision CPU control and the production default. Unsupported capability
limits and Web also remain on the CPU path.

## Exact GPU validation

A real-device unit test submits 1,024 overlapping receiver AABBs, maps the
dense coverage asynchronously, compacts it, and requires the complete ordered
result to equal `directional_receiver_demand`. The ABI-size and WGSL parser
tests run independently. Runtime telemetry in the moving 1,140-receiver
fixture reported:

- 36 dispatches and 35 completions in 39 useful frames;
- `gpu_receiver_validated: true`;
- `receiver_marking_backend: gpu-async-lagged`;
- one expected in-flight result;
- zero validation failures.

The CPU and GPU controls had identical demand count, cache hits, misses,
residency, dirty pages, invalidations, rendered pages, denials, evictions, and
fallback mode.

## Image evidence

The 1,140-receiver stress nodes are receive-only and fully below the large
ground, so the oracle exercises receiver motion without placing boxes in the
capture. GPU versus same-revision CPU control produced:

- RMSE `0.000018280`;
- SSIM `1.000000000`;
- zero pixels above the 0.02 tolerance;
- mean OKLab delta `0.000000223`;
- mean edge delta `0.000000204`.

The ordinary two-receiver VSM fixture remained on fixed CPU marking, allocated
zero GPU-marker bytes, dispatched zero marker passes, and matched the previous
milestone at RMSE `0.000006212`, SSIM `1.000000000`, with zero pixels above
tolerance.

With VSM disabled, telemetry again reported zero VSM capacity, bytes,
residency, and work. The image matched the previous milestone at RMSE
`0.000006775`, SSIM `1.000000000`, with zero pixels above tolerance.

## Rejected performance qualification

The original experimental threshold of 256 was rejected rather than shipped.
At roughly 285 moving receivers, three CPU controls and three GPU candidates
showed a small CPU/wall improvement but inflated GPU timestamp measurements
from the cost of a readback every frame. That workload therefore stays on the
fixed CPU path and allocates nothing for this feature.

The preliminary 1,140-receiver paired moving probe measured:

- wall mean: `4539.346179` to `4470.161396 ms`;
- CPU frame mean: `18.615338` to `18.306413 ms`;
- GPU frame mean: `40.886617` to `37.220795 ms`;
- shadow CPU mean: `0.637508` to `0.386642 ms`;
- virtual-page CPU mean: `0.034791` to `0.034375 ms`.

Those totals were dominated by macOS window-server throttling and by Bloom's
diagnostic profiler synchronizing GPU results. They were not sufficient to
attribute the GPU marker's own cost.

Follow-up timestamp instrumentation measured the unchanged marking dispatch
directly at `0.485908 ms`. The fixed CPU control's complete shadow-pass delta
was only `0.250866 ms`, so even the marker alone consumed more GPU time than
the CPU work it removed. The attempted GPU rank/compact follow-up measured
`0.436717 ms` and `0.504675 ms` in addition; a single-pass version measured
`1.510075 ms` for the combined work. That candidate was discarded before
commit.

This fails the project's across-the-board no-regression requirement. No
receiver-count crossover has been qualified, so there is no automatic
workload threshold. The fixed CPU oracle is the default; the retained GPU
backend is an explicit experiment for future architectures and profiling.

## Regression gates

- FFI/schema parity passed for every declared platform, including Linux and
  Web.
- Formatting, strict correctness/performance Clippy, and the file-size ratchet
  passed.
- The `wasm32-unknown-unknown` Web-feature check passed.
- 345 shared unit tests passed with 1 ignored.
- The headless negotiated-device test passed.
- 59 GPU goldens passed with 2 hardware-policy tests ignored.
- All 4 render-target tests passed.
- 38 focused VSM tests passed.
- Quality governance, visual-diff, asset-cooker, and all 20 canonical-example
  inventory gates passed.

Machine-readable measurements accompany this note in
`docs/evidence/issue-132-async-gpu-receiver-v1.json`.

The next VSM implementation work must start from a GPU-resident design that
amortizes or eliminates dispatch/readback cost and must pass direct pass-level
timing before automatic activation. GPU page-caster culling and submission
remain separate milestones.
