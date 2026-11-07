# Issue #128 host-load preflight v1 evidence

The strict Apple M1 Max / Metal rerun after baseline installation accepted all
nine visual comparisons and all applicable GPU gates, but unrelated compiler
work inflated CPU p95 enough to fail four scene budgets. Repeating those scenes
while the same external jobs were active preserved the images and GPU timing
range while making CPU p95 substantially worse. Those measurements cannot
distinguish a Bloom regression from scheduler contention.

This checkpoint makes hard-budget runs prove that the host is suitable before
they measure it:

- every machine class versions an aggregate CPU-utilization limit and a
  single-process limit;
- the runner requires three consecutive accepted samples before it builds or
  captures;
- a busy host waits for up to 120 seconds by default, with a configurable wait
  for shared workstations;
- timeout exits with code 3 and `host-preflight.json`, reported as **retry when
  idle**, rather than a renderer failure;
- every completed case writes `host-postflight.json`; if unrelated load appears
  during capture, its hard performance comparisons are suppressed and the
  measurement is explicitly invalidated;
- visual, telemetry, and resource-contract checks remain active.

The versioned threshold is 20% aggregate utilization across logical CPUs, with
no other process above 75% of one CPU. The three samples are two seconds apart.
Waiting longer changes only when measurement starts; it does not alter quality,
warm-up, measured frames, or any CPU/GPU/VRAM budget.

## Verification

- `./scripts/ci-check.sh --quick --component quality-contract`: passed;
  51 Python orchestration/corpus tests, three `bloom-diff` fault/metric tests,
  and 48 `bloom-cook` tests passed.
- Host-preflight unit coverage proves aggregate-pressure rejection,
  single-process rejection, consecutive-idle sampling, and immediate busy-host
  evidence.
- A live hard-gated invocation on the contended M1 Max exited 3 before build or
  capture, reporting 90.42% aggregate CPU utilization against the 20% limit.
  Its evidence SHA-256 is
  `e29fd42a9d1bc5715e3364a31e61a185df50020b781fcb2154c25a25583fbd73`.
- `python3 tools/quality/run.py check`: passed for all nine cases after restoring
  the exact `niagara_bistro` revision and the pinned HDR.
- `bash -n scripts/ci-check.sh`, Python bytecode compilation, and Git whitespace
  checks passed.

This checkpoint prevents contaminated measurements from being recorded as
Bloom performance regressions. It does not itself satisfy the pending strict
Metal performance gate; that still requires a host that passes the preflight.
