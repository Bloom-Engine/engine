# Issue #128 bootstrap matrix fail-closed v1 evidence

This checkpoint audits the complete nine-case quality manifest and fixes a
qualification ordering defect exposed during initial baseline bring-up. The
exact tested commit is `465c5f50d0dd95799f3d01935401f08f5df9fa92`.

## Defect and correction

Before this checkpoint, a completed capture with no approved baseline returned
from `run_case` before applying `performance_failures`. A report-only bootstrap
run could therefore record only `approved baseline missing` even when the same
telemetry exceeded a hard CPU/GPU budget. Missing-intermediate failures were
also duplicated after visual comparison.

The runner now evaluates telemetry, steady-state resources, required
intermediates, and applicable hard performance budgets before checking whether
the independent visual baseline exists. Missing baselines remain failures and
still require explicit human review/install. Report-only results are marked as
failures while continuing to return success to the exploratory caller.

A unit test exercises the exact failure ordering with an absent baseline, a
missing required intermediate, and CPU/GPU budget overruns. The real PBR case
then proved that all three independent causes are emitted together.

## Clean nine-case Metal capture

The full manifest ran from a clean exact commit on Apple M1 Max / Metal. All
nine examples built and captured their required final and intermediate
artifacts. High-preset cases emitted 30 named intermediate PNGs each; the
constrained case emitted its eight applicable graph products. Every case
reported:

- zero steady-state bind-group creation;
- zero steady-state graph compilation and first-use pipeline creation;
- zero steady-state transient physical texture or buffer creation;
- exactly one frame-submission command encoder.

Every visual failure is the expected, truthful absence of an independently
approved baseline. A complete initial review bundle was generated, but no
baseline was installed or committed because human approval is a separate
governance boundary.

## Applicable Metal budgets

The fail-closed result exposes the actual qualification work remaining instead
of hiding it behind missing baselines:

| Case | CPU p95 / budget (ms) | GPU p95 / budget (ms) | Hard performance result |
|---|---:|---:|---|
| PBR spheres high | 5.801 / 5.000 | 23.667 / 20.000 | fail CPU + GPU |
| Damaged Helmet | 5.074 / 6.000 | 11.956 / 12.000 | pass |
| Sponza interior | 5.390 / 10.000 | 57.116 / 55.000 | fail GPU |
| Skinned alpha motion | 15.707 / 10.000 | 29.772 / 30.000 | fail CPU |
| Weighted transparency | 7.766 / 12.000 | 31.927 / 30.000 | fail GPU |
| Masked alpha coverage | 4.355 / 12.000 | 21.837 / 30.000 | pass |

Bistro exterior and draw/light stress declare RTX 4080 / Vulkan budgets, so
their M1 Max measurements remain report-only and are not represented as
qualification results.

The separately selected constrained machine class was repeated three times:

| Metric | Runs (ms) | Median | Budget |
|---|---|---:|---:|
| CPU p95 | 5.094, 4.552, 3.855 | 4.552 | 5.000 |
| GPU p95 | 7.099, 6.882, 7.235 | 7.099 | 8.000 |

Two of three runs passed both hard limits. The first CPU sample exceeded its
limit by 0.094 ms; the median remains inside the existing budget. No budget or
noise bound is changed by this checkpoint.

The dominant timestamped work in the remaining GPU failures is bloom: 10.192
ms mean in PBR high and 25.669 ms mean in Sponza. Weighted transparency reports
9.218 ms bloom plus 4.045 ms weighted composition. This identifies the next
performance investigation without yet claiming a root cause or optimization.

## Validation

- focused missing-baseline/runtime-order regression test: passed;
- quality-runner tests: 13 passed;
- complete quality contract: 48 Python governance tests, three `bloom-diff`
  tests, and 48 cooker tests passed;
- clean exact-commit nine-case capture: nine of nine capture contracts
  completed; zero missing required intermediates;
- constrained qualification: three complete exact-commit runs;
- Python compilation and Git whitespace checks passed.

This is a qualification-infrastructure checkpoint, not approval of the initial
visual baselines and not closure of issue #128. Next work is to resolve or
qualify the exposed Metal budget overruns, obtain human review of the nine
candidate images, install the approved baseline set explicitly, and run the
same corpus on the required Vulkan hardware.
