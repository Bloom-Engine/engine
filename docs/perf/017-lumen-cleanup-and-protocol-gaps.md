# 017 — Lumen rollout cleanup: regression-guard + scope-match gaps

**Effort:** 1-2 days · **Expected gain:** protocol compliance, not FPS · **Status:** open

Follow-up to the Lumen roadmap (007-prep / 007a / 007b / 013 / 014 / 016). The
main acceptance target landed — `--quality 3 --ssgi 1 --fps-only 300` hits
60 fps vs the 23.6 fps baseline — but four specific gaps were found during
a post-rollout verification pass and are not tracked anywhere else.

## Problems

### 1. Regression guard fails

`docs/perf/README.md` makes `./main --quality 0 --fps-only 60` a hard
regression guard: at the "off" preset (no shadows, no post-FX) the frame must
fit the 16.7 ms vsync budget. It doesn't:

| Command | Result | Expected |
|---|---|---|
| `BLOOM_NO_FULLSCREEN=1 ./main --quality 0 --fps-only 60`  | **2.0 fps** (502 ms/frame) | ≥ 60 fps |
| `BLOOM_NO_FULLSCREEN=1 ./main --quality 0 --fps-only 300` | **31.9 fps** (31.4 ms/frame) | ≥ 60 fps |

The 60-frame run is pathologically worse than the 300-frame run, which
suggests warmup cost (BLAS build / per-mesh SDF bake / Mesh Card capture)
is running even at `--quality 0`. Either:

- Quality 0 must gate off the entire Lumen pipeline (probe placement, SDF
  bake queue, card capture queue, accel-structure build), OR
- Warmup work must be amortised so it can't dominate the first 60 frames.

The ticket closures for 007a / 007b / 013 / 014 / 016 all claim "followed the
standard perf-ticket protocol" but none of them ran the regression guard
post-landing — or the guard would have caught this.

### 2. `--ssgi 0` explicit disable path is buggy

`BLOOM_NO_FULLSCREEN=1 ./main --quality 0 --ssgi 0 --fps-only 300` measured
**0.1 fps (16 196 ms/frame)** on one run and 13.3 fps at 30 frames on
another. Inconsistent across runs, and 0.1 fps strongly suggests a hang /
readback stall on a code path taken only when SSGI is explicitly off.

Probable causes to investigate:

- A probe / SDF / card buffer is still allocated and cleared every frame
  even when SSGI is off.
- A dispatch with zero workgroups producing an implicit wait.
- A deadlock-y fallback in the HW path when the feature is off but the
  TLAS rebuild still runs.

### 3. Ticket 016 scope does not match what was implemented

016's problem statement and approach section describe two specific
techniques:

- **CDF-based importance sampling**: precomputed row-sum + row-CDF of the
  prev-frame probe radiance SIM, inverse-CDF sampled per ray, per-ray PDF
  weight tracked for unbiasedness.
- **Hierarchical refinement**: a second probe layer at 8-pixel stride for
  high-variance tiles only, resolve-pass preference for the fine layer.

Neither shipped. V1-V4 instead landed:

- V1 per-frame R2 octahedral jitter.
- V2 per-probe decorrelated jitter.
- V3 prev-frame-aware jitter magnitude scaling.
- V4 variance-adaptive temporal EMA.

The closure note ("the ticket's CDF-based importance sampling + separate
refinement probe layer were not needed at the 64-rays/probe shape we already
ship; V4's adaptive alpha captures the same 'high-variance regions need more
attention' intent") is a scope change, not a completion. Decide between:

1. **Rewrite 016** to describe the jitter-and-variance-EMA approach that
   shipped, mark landed, and close. Open a follow-up ticket if
   CDF-sampled importance + hierarchical refinement are still wanted.
2. **Re-open 016** and implement the original approach. Lumen's published
   quality-per-ray wins come specifically from prev-frame PDF sampling;
   variance EMA converges disocclusions faster but doesn't shift ray
   directions toward bright incoming light. The two aren't interchangeable.

Either path is defensible; pick one and make the docs match the code.

### 4. Uncommitted / untracked state

- `native/web/src/lib.rs` has uncommitted changes: adds `bloom_is_initialized`
  and an `INIT_STARTED` guard against double wgpu init when both the JS
  orchestrator and Perry's `main()` call `bloom_init_window`. Looks
  correct; commit or discard with a clear reason.
- `.claude/scheduled_tasks.lock` deleted, not committed. Likely harness-local
  noise but should be decided on.
- `examples/intel-sponza/assets/` untracked. Are they part of the
  benchmark or a gitignore'd local-only cache? Clarify.
- `tools/{babylonjs,unity,unreal,threejs}_reference/`, `tools/dump_dds/`
  untracked. These exist from earlier work; decide whether to commit
  (with README explanations), add to `.gitignore`, or delete.

## Acceptance

- `./main --quality 0 --fps-only 60` ≥ 60 fps (vsync).
- `./main --quality 0 --ssgi 0 --fps-only 300` ≥ 60 fps, deterministic
  across at least three consecutive runs.
- `./main --quality 3 --ssgi 1 --fps-only 300` still ≥ 47.2 fps (no
  regression on the main target — currently vsync-capped 60 fps).
- 016 either rewritten to match shipped V1-V4 or re-opened with a concrete
  V5 plan for CDF sampling + hierarchical refinement.
- `git status` clean from the repo root (no uncommitted diffs, all
  untracked tool-reference dirs resolved).

## Files likely to change

- `native/shared/src/renderer/mod.rs` — gate probe / SDF / card passes on
  effective-SSGI-enabled state at the *frame-graph* level, not inside
  shaders. Likely also where the `--ssgi 0` hang lives.
- `native/shared/src/engine.rs` — if quality-preset gating of Lumen
  warmup queues is missing, it belongs here.
- `native/web/src/lib.rs` — commit or revert the init-guard change.
- `docs/perf/016-lumen-importance-sampling.md` — rewrite or re-open.
- `docs/perf/README.md` — 017 row in the Lumen GI table.
- `.gitignore` — if any of the untracked dirs should stay local.

## Notes for the implementer

- The regression-guard failure is the load-bearing finding. Fix it first;
  the other three are bookkeeping on top.
- Don't "fix" the guard by changing the 60 fps threshold in the protocol
  doc. Root-cause whatever quality-0 work isn't gated.
- Per the doc, do not trust per-pass GPU timestamps on Metal for small
  fullscreen passes. Use `--fps-only` as ground truth for accept/reject.
- `BLOOM_FORCE_SW_GI=1` exists and forces the SW path even on RT-capable
  adapters — useful to isolate whether the `--ssgi 0` hang is on the HW
  side only.
