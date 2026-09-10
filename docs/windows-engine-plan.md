# Engine completion plan

User objective: work through the plan until all of it is finished. The working
host is Windows with an AMD Radeon 760M; an RTX 4080 is unavailable. This plan
preserves both the immediate qualification sequence and the unfinished engine
areas discussed on 2026-09-10. A completed local test does not close a broader
platform, packaging, usability, or hardware requirement.

The existing renderer work is in draft PR #147. Windows execution fixes and the
first nine-scene Radeon evidence are in draft PR #154. Follow-up work starts at
`e8e08b90fc7cbad3e060d5ddb5c3ddb87aeb7add` on `codex/windows-engine-plan`.

## Qualification and integration sequence

| Work | Required completion evidence | Current state |
| --- | --- | --- |
| #127 Vulkan PT correctness | Three deterministic progressive and motion runs, both negative controls, finite intermediates, reset/lighting/rigid-motion checks, retained report | Canonical hardware gate and all four focused temporal tests pass on Radeon/Vulkan; report pending |
| #128 Windows image discrepancies | Identify the first incorrect stage or document a reviewed backend-specific baseline decision; rerun the full strict corpus and reproducibility checks | Sponza and skinned/alpha still fail against portable baselines; original baseline source reproduces both failures |
| #135 / #149 temporal reconstruction | Enforced motion/producer/quality-preset corpus, representative scenes, fractional/native and frozen A/B timing, memory/resize checks, platform evidence | Initial wider Windows run exposed a Vulkan counter defect and false GPU-unavailable skips after repeated device creation; no broad pass claimed |
| #140 integration gates | Same required local/hosted lanes pass on exact source; release package startup and all-example evidence | #147 contract ratchet fails in nine files; Android/iOS/tvOS/visionOS jobs fail on stale lockfiles before compilation |
| #138 capability fallback | Actual constrained-adapter startup and relevant forced-tier corpus, truthful capability outputs | Existing implementation/evidence preserved; physical constrained-limit acceptance still needs proof |
| PR integration | Reviewable changes, passing required checks, full issue evidence, merge-ready rendering branch | #147 and #154 remain drafts; no merge performed |

## Engine work retained in scope

Each linked issue's complete design, acceptance criteria, compatibility contract,
and verification commands remain required. The issue snapshots used for this
audit are saved in `tools/quality/out/windows-engine-plan/plan-requirements.json`.

| Work | Remaining outcome |
| --- | --- |
| #27 visibility shading | Full compatibility/effects/occlusion corpus; admitted-workload performance improvement without low-overdraw regression; integrated and discrete evidence before activation |
| #131 virtual geometry | Required discrete-adapter motion qualification with no holes, cracks, page flashes, or unbounded trails; preserve existing integrated-GPU evidence |
| #148 shared glTF instances | Deterministic placement/bounds/material verification, mirrored-transform and fallback coverage, full images and uncapped timing |
| #137 asset/world streaming | Async state machine, priorities, cancellation, bounded CPU/GPU memory and uploads, slow-IO/churn tests, HLOD/world activation, native/web/mobile paths |
| #141 API generation | One schema governing manifests/Rust/TypeScript/docs, safe collections and packed buffers, UTF-8, generated parity and migration coverage |
| #142 starter/examples, #74 startup | One-command native and browser starter, all canonical examples compiled, real startup/render smoke, validated manifests, actionable setup failures |
| #143 component facade | Optional components and lifecycle, deterministic fixed update, prefab round trip, ownership/cleanup and async-destruction safety, 10k-entity profile |
| #144 runtime UI | Retained layout/clip/scroll, focus and pointer capture, mouse/touch/keyboard/gamepad navigation, UTF-8 entry, accessibility/capabilities, DPI and 1k-widget tests |
| #145 packaging | Versioned shader-runtime dependencies, clean Windows install and shader compilation, missing-dependency behavior, UTF-8 round trips, validated platform packages |
| #153 original hardware contract | RTX 4080-specific strict qualification remains unperformed; Radeon observations do not satisfy this named hardware requirement |

## Completion discipline

- Preserve approved images, thresholds, existing performance budgets, and noise
  bounds unless a separate reviewed change is justified.
- Require the intended adapter/backend and reject test infrastructure failures;
  a silently skipped supported GPU test is not evidence of completion.
- Keep raw commands, exact source identity, outputs, captures, and artifact hashes
  with each qualification report.
- Continue work that this machine can perform while hardware-specific items wait.
- Mark the overall plan complete only after every applicable issue requirement
  above is proven against current code and external state.
