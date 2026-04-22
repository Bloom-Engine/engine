# 017 — Lumen rollout cleanup: regression-guard + scope-match gaps

**Effort:** 1-2 days · **Expected gain:** protocol compliance, not FPS · **Status:** landed

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

## Landed state

### Root cause — #1 and #2 were the same bug

The Lumen warmup + per-frame block in `Renderer::end_frame_with_scene`
(renderer/mod.rs around the original line 7228-7277) ran unconditionally:

- `rebuild_acceleration_structures` (BLAS/TLAS refresh)
- `capture_pending_mesh_cards` (rate-limited 20 slots/frame — ~120 frames
  to drain Sponza's 2430-slot backlog)
- `bake_pending_sdfs` (8 per-mesh UDFs/frame)
- `maybe_invalidate_sdf_clipmap` + `bake_scene_sdf_clipmap`
- `maybe_invalidate_wsrc` + `bake_wsrc` (per cascade)
- `light_mesh_cards` (per-frame card relight compute — runs every frame
  once any card slot is populated, which is all frames after first)

Every one of these feeds the SSGI probe trace downstream. The probe trace
itself was gated on `self.ssgi_enabled`, but the warmup + relight passes
weren't. At `--quality 0` the card-capture backlog + per-frame relight
compute held the frame 3-500 ms depending on where in the drain schedule
you measured — hence the 2.0 fps vs 31.9 fps vs 27.6 fps numbers that
moved across runs. `--ssgi 0` hit the same pattern.

### Fix — one conditional block in the renderer

Wrapped the whole Lumen warmup + relight section in
`if self.ssgi_enabled { ... }`. Pending queues stay populated, so a
runtime `setSsgiEnabled(true)` resumes baking from where it paused.

### FPS-harness fallback

The old `FPS = getFPS()` in `examples/intel-sponza/main.ts` uses the
engine's 1-second rolling window. A 60-frame run at 60 fps finishes in
exactly 1.0 s and the sampler reset threshold (`>= 1.0 s`) can fire
either side of frame 60 — so the harness prints `0.0 fps (0.00 ms/frame)`
on fast runs. Added a cumulative `measureAccumS` (summed `getDeltaTime()`)
fallback so `--fps-only 60` at pass-speed now reports the real fps.
Doesn't change the engine's rolling-window metric — only the harness'
short-run output.

### Measurements (3× consecutive runs, BLOOM_NO_FULLSCREEN=1)

| Command | Before | After |
|---|---|---|
| `--quality 0 --fps-only 60`  | 2.0 fps     | **65.7 / 65.4 / 65.0 fps** ✅ |
| `--quality 0 --fps-only 300` | 31.9 fps    | **59.9 / 60.2 / 60.0 fps** ✅ |
| `--quality 0 --ssgi 0 --fps-only 300` | 0.1 fps / 13.3 fps | **60.0 / 60.1 / 60.0 fps** ✅ |
| `--quality 3 --ssgi 1 --fps-only 300` | 60.0 fps | **60.0 / 60.2 / 59.9 fps** ✅ |

All four acceptance conditions pass.

### #3 — Ticket 016 rewrite

Rewrote 016 top-to-bottom. Problem + Approach now describe the shipped
V1-V4 temporal-jitter + variance-adaptive EMA work. Deferred original
scope (CDF importance sampling + hierarchical refinement probe layer)
pointed at the pre-existing GitHub issues
[#19](https://github.com/bloom-engine/bloom/issues/19) and
[#20](https://github.com/bloom-engine/bloom/issues/20). The V4 closure
table at the bottom of 016 was already present from the 9af6111 commit;
left intact so the ticket reads "here's what shipped, here's the per-V
audit underneath it."

### #4 — Untracked state

- `native/web/src/lib.rs` init-guard (`bloom_is_initialized` +
  `INIT_STARTED`) committed. Looked correct — idempotent
  `bloom_init_window` prevents the JS orchestrator + Perry `main()`
  both kicking wgpu init.
- `.claude/scheduled_tasks.lock` deleted, added to `.gitignore`
  (harness-local).
- `examples/intel-sponza/assets/outdoor.hdr` gitignored
  alongside the existing `NewSponza_Main_glTF_003.*` + `textures/`
  entries (consistent with the other Sponza assets pattern).
- `tools/{babylonjs,threejs,unity,unreal}_reference/` gitignored with
  a comment explaining they're local external-engine reference clones
  for side-by-side perf comparison, not repo source.
- `tools/dump_dds/target/` + `tools/dump_dds/Cargo.lock` gitignored;
  tool source (main.rs + Cargo.toml) committed as a small utility for
  inspecting compressed textures.
- Jolt physics WIP (`.gitmodules`, `native/shared/build.rs`,
  `native/shared/src/jolt_sys.rs`, `native/shared/src/lib.rs`
  `#[cfg(feature = "jolt")]` line, `native/shared/Cargo.toml` feature
  flag + cmake build-dep, `native/third_party/JoltPhysics` +
  `native/third_party/bloom_jolt`, `src/physics/index.v2.ts`) is a
  separate dev track and was intentionally not touched by 017 — the
  original 017 problem statement explicitly flagged it as "not perf."
  Resolution stays with that physics track, not this perf-rollout
  cleanup.
