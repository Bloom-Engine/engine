# Issue #149 fractional/native throughput v1 evidence

This checkpoint governs the remaining issue #149 performance acceptance
criterion: quality-preset-4 temporal reconstruction at render scale 0.75 must
retain a measurable end-to-end advantage over otherwise matched native 1.0.
The exact tested implementation commit is
`27217f305f0f740c81fd15a663133eabdcc5dc5a`.

## Governed comparison

The ignored real-GPU test
`quality_presets::profile_fractional_taa_native_advantage` renders a matched
1600x900 glossy-detail fixture with TAA enabled. Each run uses 120 warm-up
frames and 600 measured moving-camera frames. Three fractional/native pairs
alternate order (`F/N`, `N/F`, `F/N`) to reduce ordering and thermal bias.

The test records end-to-end wall time as the authoritative throughput measure
and timestamped GPU pass work as an independent device-side measure. GPU query
readback is deliberately serialized every measured frame for deterministic
profiling; both scales pay the same diagnostic cost. The governed threshold is
at least 5% advantage in the median of the three run means for both measures.

| Scale/run | Wall mean (ms) | Wall p50/p95 (ms) | GPU work mean (ms) | GPU p50/p95 (ms) | TAA GPU mean (ms) |
|---|---:|---:|---:|---:|---:|
| 0.75 / 1 | 5.636 | 4.895 / 10.112 | 5.125 | 5.583 / 6.084 | 1.866 |
| 0.75 / 2 | 4.900 | 4.720 / 6.332 | 5.325 | 5.621 / 6.116 | 1.939 |
| 0.75 / 3 | 5.852 | 5.005 / 11.573 | 5.254 | 5.623 / 6.089 | 1.905 |
| 1.0 / 1 | 6.432 | 5.808 / 12.033 | 6.511 | 7.474 / 8.026 | 2.385 |
| 1.0 / 2 | 6.093 | 5.881 / 8.356 | 6.885 | 7.566 / 8.117 | 2.521 |
| 1.0 / 3 | 6.618 | 5.893 / 12.384 | 6.816 | 7.576 / 8.169 | 2.491 |

| Governed metric | 0.75 median | Native 1.0 median | Advantage | Limit |
|---|---:|---:|---:|---:|
| End-to-end wall mean | 5.636 ms | 6.432 ms | 12.37% | >= 5% |
| Timestamped GPU work mean | 5.254 ms | 6.816 ms | 22.91% | >= 5% |

The exact run passed on an Apple M1 Max Metal adapter. A preceding independent
full component run also passed, measuring 9.69% wall-time advantage and 23.34%
timestamped-GPU-work advantage. The repeated result reduces the likelihood
that the governed conclusion is an isolated scheduling or thermal sample.

## Resource and CI contract

All six measured runs report zero steady-state bind-group creation, render
graph compilation, first-use pipeline creation, and transient physical texture
or buffer creation. The single per-frame command encoder is expected.

The `fractional-native-throughput` hardware component is wired into both Metal
and Vulkan quality jobs. It writes `result.json` and `summary.md` before its
assertions so failed qualification runs remain diagnosable. This checkpoint
adds test and CI governance only; it does not alter production renderer math,
passes, resources, or memory cost.

## Validation

- exact governed hardware component: passed, 12.37% wall and 22.91% GPU-work
  advantage;
- independent preceding component run: passed, 9.69% wall and 23.34% GPU-work
  advantage;
- quality-preset real-GPU subset: 11 passed, zero failed, two explicit profile
  tests ignored;
- quality contract: 47 Python governance tests, three `bloom-diff` tests, and
  48 cooker tests passed;
- Rust formatting, Clippy under Rust 1.98, shell syntax, workflow lint, and CI
  contract checks passed;
- GitHub quality-contract job at prerequisite commit `e3b9446` passed.

This satisfies the fractional-versus-native performance criterion on the
tested Metal qualification GPU. It does not close issue #149: a hosted Vulkan
qualification result and the rest of the representative platform matrix remain
outstanding. At evidence time the repository has zero registered self-hosted
Actions runners, so no Vulkan hardware result can be claimed.
