# Windows GPU profiler integrity

## Result and scope

The Windows Radeon 760M/Vulkan profiler returned the previous frame's query
values, including stale buffer contents in the first frame. Submitting the
readback copy after the renderer's query-resolve submission repairs that
behavior. An independent resolve of the same query events matches the reported
current frame from its first sample.

This is observed with wgpu 29.0.1 and AMD driver 24.12.1 (LLPC). The evidence
identifies resolve-to-copy visibility/order behavior on this configuration; it
does not establish an upstream driver or library defect. The fix retains the
existing blocking map and adds one profiler-only copy submission. Normal
rendering with profiling disabled is unaffected.

## Diagnosis and controls

The initial timing-only regression records four frames with each of three fresh
profilers. Against production source `64d5eed`, all 12 samples mismatch the
independent query values. For example, expected frame durations start at
53.72, 53.96, 68.52, and 53.84 microseconds; the old profiler reports 0, 53.72,
53.96, and 68.52. Recreating the profiler can reuse the preceding instance's
last buffer contents. The fixed implementation passes the same comparison.

Diagnostic captures distinguish the stages:

- Mapping and polling both report success even when the normal readback is stale.
- A later independent copy of the resolve buffer has the current query values.
- Re-reading the normal readback retains the old values.
- Explicit zero initialization only makes the first bad sample zero; it does
  not remove the one-frame delay.
- Moving the copy into the following queue submission makes the two captures
  equal without an additional CPU wait.

`current-frame-before.log` and `current-frame-before-test.rs` retain the initial
negative control. The final GPU regression also verifies reporting coverage
and rejects a reserved but unresolved frame. Temporary diagnostic instrumentation
is retained in the evidence directory and is absent from production.

## Reporting changes

Map or poll failures no longer read an unmapped buffer. Missing resolves,
backwards query pairs, invalid timestamp periods, and exhausted query budgets
produce incomplete frames. Rolling pass means drop unavailable latest samples
and no longer reuse overwritten ring entries or average missing samples as zero.

The native report distinguishes timestamp capability from valid measurement
coverage using `timing_window_frames`, `gpu_timing_valid_frames`, and
`gpu_timing_valid`. GPU aggregates exclude incomplete frames; the qualification
runner rejects an incomplete or unverified window. The window is the most recent
120 frames at most, not an all-session percentile claim. Non-finite timing,
timestep, render-scale, and hard-budget numbers also fail qualification.
Existing telemetry without coverage fields needs recapture for timing acceptance.

## Validation

The complete shared release suite passed on the physical Radeon/Vulkan adapter:
487 library tests, 90 golden tests, device startup, four focused PT temporal
tests, and the remaining enabled runtime/invariance suites. There were no test
failures. Existing exclusions remain: one library ignore, four golden ignores,
two documentation ignores, and external-input early returns (two Bistro golden
cases plus detailed virtual geometry, large virtual stress, and full Bistro
visibility fixtures). These returns do not qualify those external scenes.

The repository contracts, lint lane, and quality-contract lane also passed,
including 25 quality-runner unit tests. The current-frame GPU regression also
passes on the Radeon through DX12. The regression compares actual query
events instead of imposing a GPU-duration threshold. The new SSGI measurement
assertion requires all 120 measured GPU frames to be complete.

The hosted macOS shared lane is green, but its timestamp regression's `ok`
status alone does not prove GPU execution: the fixture returns early when
timestamps are unavailable. A subsequent native capture on the hosted Apple
Paravirtual device reports no timestamp capability. Metal GPU timestamp
qualification remains unproven. The initial release notes and shadow archive
README overstated that coverage; this clarification supersedes those claims.

## Corrected SSGI comparison

Both frozen executables use the corrected profiler and identical fixture code.
Only the two SSGI shader source files differ, using the same before/after shader
snapshots as the earlier stationary-SSGI report. Each resolution has five
alternating process pairs, with 24 warm-up and 120 measured frames. No local
build or other GPU test runs during the measurement sequence. The reported
cost is the sum of the five probe passes; blocking instrumentation means these
figures do not establish uninstrumented frame throughput.

| Resolution | Before median (range), ms | After median (range), ms | Median delta |
| --- | --- | --- | --- |
| 256 | 0.098082 (0.097440-0.101088) | 0.106680 (0.100892-0.107854) | +0.008598 ms |
| 1280x720 | 0.396267 (0.387516-0.397251) | 0.439799 (0.431261-0.440850) | +0.043532 ms |

The old shader's stationary-image assertion still fails; these failures are
retained as controls. The fixed shader passes, and all timing windows on both
sides contain 120 complete frames. The earlier SSGI timing table predates this
profiler correction and is superseded by this comparison. Its image evidence
and the outstanding HD TAA-jitter failure are independent of timestamp readback.

## Evidence and remaining work

Raw commands, logs, source snapshots, frozen executables, timing records, and
SHA-256 receipts are published with the
[profiler evidence prerelease](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-profiler-windows-vulkan-20260910).
The source branch is `codex/windows-profiler-integrity`, based on #155 at
`64d5eedd71e4249edfef8977af9c0e4a944753d3`; the release tag identifies the final
report commit and its archive manifest hashes every retained file.

This closes the observed current-frame readback defect on the tested adapter.
The two Windows portable-image discrepancies, HD temporal stability, the
representative scene/performance corpus, and other hardware/platform acceptance
remain open. No image baseline, visual threshold, or performance budget changes.
