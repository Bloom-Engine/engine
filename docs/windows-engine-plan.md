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
| #127 Vulkan PT correctness | Three deterministic progressive and motion runs, both negative controls, finite intermediates, reset/lighting/rigid-motion checks, retained report | Canonical hardware gate, all four focused temporal tests, and CPU reference sanity check pass on Radeon/Vulkan; [report](evidence/issue-127-windows-vulkan-v1.md) and [raw evidence](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-155-windows-vulkan-20260910) published |
| #128 Windows image discrepancies | Identify the first incorrect stage or document a reviewed backend-specific baseline decision; rerun the full strict corpus and reproducibility checks | [Cutout phase correction](evidence/windows-alpha-phase-v1.md) passes all nine Radeon image gates in two complete runs and both hosted Metal images. Reproducibility passes with 257 byte-identical artifacts. Both strict runs fail postflight host-load checks, so their timing remains unqualified |
| #135 / #149 temporal reconstruction | Enforced motion/producer/quality-preset corpus, representative scenes, fractional/native and frozen A/B timing, memory/resize checks, platform evidence | Device/resource, stationary SSGI, and profiler fixes are retained. The [surface reconstruction correction](evidence/windows-ssgi-surface-v1.md) passes the original HD startup limits on Radeon/Vulkan; all 93 local goldens that run pass, including lighting recovery. Full scene-image, timing, and platform qualification of that correction remain open, as does the wider representative corpus |
| #140 integration gates | Same required local/hosted lanes pass on exact source; release package startup and all-example evidence | All 24 hosted checks pass at #158 source `662a44f`, including macOS shared/golden tests, mobile target builds, native/web builds, browser startup, and canonical Metal images. Scheduled physical-hardware checks, all-example compilation, and release-install acceptance remain separate requirements |
| #138 capability fallback | Actual constrained-adapter startup and relevant forced-tier corpus, truthful capability outputs | Existing implementation/evidence preserved; physical constrained-limit acceptance still needs proof |
| PR integration | Reviewable changes, passing required checks, full issue evidence, merge-ready rendering branch | #147 and the stacked fixes #154–#158 remain drafts; no merge performed |

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

## Current next steps

1. #155 source `64d5eed` has passing hosted checks and published evidence. The
   archive SHA-256 is `de9c1beca73bfcf60bf79d3a612b72c072f726f6272576a7f0f60ffdbc7ce25d`.
   CI run URLs and conclusions are in its separate `pr155-checks-64d5eed.json`
   release asset. No draft PR has been merged.
2. The [profiler correction](evidence/windows-profiler-integrity-v1.md) on
   `codex/windows-profiler-integrity` passes local contracts, lint, the quality
   lane, and the complete shared suite. Its current-frame regression rejects
   all 12 old Vulkan samples and passes with the correction on Vulkan and DX12.
   Corrected SSGI timing covers 20 isolated runs, each with 120 complete GPU
   frames. Hosted Metal's shared lane passes, but its Apple Paravirtual adapter
   lacks timestamp queries: the profiler GPU regression explicitly skips.
   The retained `--nocapture` log at `1965a0b` confirms this; a test reported as
   "ok" after that early return does not qualify Metal GPU timing.
   Its colored-shadow
   failure exposed an [inverse-matrix upload defect](evidence/windows-transmitted-shadow-inverse-vp-v1.md);
   the correction passes the isolated local check and rejects the wrong-color
   control. At follow-up source `fa93690`, all 23 hosted checks and the complete
   local shared suite pass. The shadow regression passes on Metal. The shadow archive
   and CI receipt are published alongside the immutable profiler archive.
3. Diagnose Sponza and skinned/alpha against the portable baselines, then repair
   HD temporal stability and complete the representative temporal/geometry
   corpus. Recapture affected timing evidence with explicit coverage fields.
   Disabling foliage shadow casting retains the Windows skinned/alpha mismatch;
   canonical captures at `98cce62` pass on hosted Metal with SSIM 0.997442544 for
   Sponza and 0.999417603 for skinned/alpha. That Apple Paravirtual adapter uses
   the modern tier and software GI and exposes no timestamps. At `0dd8f67`,
   [PR #157](https://github.com/Bloom-Engine/engine/pull/157) has all 24 hosted
   checks passing; both Metal images pass with raw export enabled. The
   [published diagnostic archive](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-image-portability-20260910)
   retains exact depth and MRT bytes, commands, checksums, and source identity.
   On matching modern/software-GI paths, Windows still fails both images.
   Skinned/alpha has 9,745 depth coverage disagreements before TAA, while
   albedo RGB closely agrees on matching surfaces. Disabling foliage shadows
   and an isolated isotropic alpha-sampling control retain the failure.
   Exact cutout-input probes at `30e7625` identify a different Bayer phase
   extent: Metal's observed extent predicts every inspected threshold away from
   integer LOD boundaries. The [integer phase correction](evidence/windows-alpha-phase-v1.md)
   preserves that approved grid on both backends. Focused Windows images now
   pass at SSIM 0.986160457 and 0.990073442. At `662a44f`, all nine Radeon image
   gates pass twice and reproducibility passes with 257 byte-identical artifacts.
   Both full runs fail their unchanged postflight host-load checks: System CPU
   exceeds the 75% per-process limit on Bistro, and in the second run also on
   draw/light stress and weighted transparency. Those timing windows remain
   unqualified. The [#158 evidence archive](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-alpha-phase-20260910)
   retains both failed strict runs and their separate passing image/repro checks.
   All 24 hosted checks pass; the new cutout GPU regression actually executes on
   Vulkan, DX12, and Metal. Raw export leaves unmodified Windows final PNGs byte-identical.
   Shared-runner timing cannot qualify hardware budgets.
   The HD TAA fixture originally failed at its required 16-frame warm-up and
   passed diagnostic controls at 32, 64, and 128 frames. The follow-up
   [surface correction](evidence/windows-ssgi-surface-v1.md) repairs depth-texel
   coordinates and resolution-dependent normal reconstruction. It passes the
   original 16-frame HD limits, removes the observed horizontal GI bands, and
   preserves the existing lighting-recovery regression on Radeon/Vulkan.
   Both isolated partial corrections fail. Full scene and platform checks of
   the combined change remain in progress; no threshold or warm-up was relaxed.
4. Continue starter/all-example and release-install checks, asset/world streaming,
   schema-generated APIs, components, and runtime UI against each issue's full
   acceptance criteria. Hardware-specific acceptance remains open while local
   work progresses.

The immediate order is to repair the HD temporal startup failure, obtain valid
full Radeon timing windows, and finish all-example startup and clean Windows
installation checks. Then complete the wider temporal/geometry corpus and the
engine API, streaming, component, and UI requirements above. These remaining
outcomes include both implementation work and acceptance evidence; they are not
a claim that each subsystem is absent. None requires waiting for an RTX 4080 to
continue local work.
