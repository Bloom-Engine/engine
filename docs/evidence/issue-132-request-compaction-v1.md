# Issue #132 receiver request-compaction qualification v1

This checkpoint qualifies the fixed-address CPU request oracle at revision
`b4e96f0dd4a30537448dfd2dd8a0f33c06a7fc5b` on an Apple M1 Max using
Metal and Bloom's native high-end profile.

## Qualified behavior

Directional receiver coverage previously used a hash map per clip level even
though every level has exactly 32 by 32 virtual addresses. The new oracle uses
one 1,024-entry `u32` coverage domain, matching a 4,096-byte R32 GPU storage
buffer. It also records only first-touched addresses, so sparse scenes rank
their touched pages rather than scanning every empty counter.

Coverage saturation, center-distance tie breaking, page coordinates,
per-level caps, and near/mid/far interleaving are unchanged. A retained test
runs the previous hash implementation against overlapping, outside, invalid,
and duplicate bounds and requires the complete ordered demand vector to match.

This is deliberately not presented as GPU marking yet. It establishes the
bounded reference algorithm and buffer ABI without a same-frame readback,
extra render pass, persistent allocation, or asynchronous frame lag.

The runtime contract remains:

- 1,024 possible addresses per level;
- per-level selected caps of 144, 64, and 16;
- at most 224 compacted directional requests;
- unchanged deterministic request order and cache ownership;
- missing or omitted pages sample CSM.

## Image and telemetry evidence

Exact-revision comparisons against the prior hash oracle were effectively
identical:

- stationary dynamic fixture: RMSE `0.000009371`, SSIM `1.000000000`, zero
  pixels above tolerance;
- moving-light transition: RMSE `0.000013946`, SSIM `1.000000000`, zero
  pixels above tolerance;
- VSM-disabled ordinary fixture: RMSE `0.000007557`, SSIM `1.000000000`,
  zero pixels above tolerance.

The stationary and moving-light VSM counters matched their controls exactly.
The disabled run reported zero VSM capacity, memory, residency, and work.
The opt-in allocation remains fixed at 19,951,824 bytes because compaction
adds no persistent renderer resource.

## Performance evidence

The moving-light oracle recomputes receiver coverage every 30 frames and
therefore exercises request compaction during its measured window. Three runs
before and three runs after the change used 120 warmup and 120 measured frames,
a fixed 60 Hz timestep, quality preset 3, and native render scale.

Every median moved down:

- wall time: `14.385408` to `12.873031 ms`;
- shadow CPU: `0.218197` to `0.181323 ms`;
- shadow GPU: `2.969117` to `2.864068 ms`;
- virtual-page CPU: `0.044228` to `0.034917 ms`;
- virtual-page GPU: `2.541226` to `2.505326 ms`.

The renderer change is CPU-only, so GPU and wall deltas include normal system
variance. The directly affected CPU medians improved by `0.036874 ms` for the
whole shadow pass and `0.009311 ms` for bounded virtual-page work. Stable
frames retain their prior demand-cache fast path.

## Regression gates

- FFI/schema parity passed for every declared platform, including Linux and
  Web.
- Formatting, strict correctness/performance Clippy, and the file-size ratchet
  passed.
- Native release compilation and the `wasm32-unknown-unknown` Web-feature
  check passed.
- 342 shared unit tests passed with 1 ignored.
- The headless negotiated-device test passed.
- 59 GPU goldens passed with 2 hardware-policy tests ignored.
- All 4 render-target tests passed.
- 35 focused VSM tests passed.
- All 20 canonical examples passed their inventory gate.

Machine-readable measurements accompany this note in
`docs/evidence/issue-132-request-compaction-v1.json`.

The next milestone is capability- and workload-gated GPU marking/compaction
that consumes this ABI without a blocking readback. GPU page-caster culling and
submission follow separately.
