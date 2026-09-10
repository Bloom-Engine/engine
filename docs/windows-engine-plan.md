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
| #128 Windows image discrepancies | Identify the first incorrect stage or document a reviewed backend-specific baseline decision; rerun the full strict corpus and reproducibility checks | Cutout and surface corrections pass all nine Radeon images. At #159 source `d610d6a`, full runs 2 and 3 pass every configured check and reproduce 257 artifacts byte-identically with matching metadata and timing differences inside existing noise bounds. Earlier invalid runs retain their failures; named hardware acceptance remains separate |
| #135 / #149 temporal reconstruction | Enforced motion/producer/quality-preset corpus, representative scenes, fractional/native and frozen A/B timing, memory/resize checks, platform evidence | Device/resource, stationary SSGI, and profiler fixes are retained. The surface correction passes original HD startup limits and 154,720 analytic receiver checks on Vulkan, DX12, and hosted Metal; 93 local goldens pass, including lighting recovery. The full Radeon corpus passes twice. Wider representative scenes, frozen A/B performance, memory/resize, and platform acceptance remain open |
| #140 integration gates | Same required local/hosted lanes pass on exact source; release package startup and all-example evidence | #159 has 24 successful CheckRun statuses, but Windows logs reveal no Cargo execution because Bash was invoked through PowerShell without an explicit shell. Local Windows execution is verified. This follow-up selects Bash and requires execution-summary artifacts; actual hosted validation is pending. All-example PR compilation and release-install acceptance remain separate requirements |
| #138 capability fallback | Actual constrained-adapter startup and relevant forced-tier corpus, truthful capability outputs | Existing implementation/evidence preserved; physical constrained-limit acceptance still needs proof |
| PR integration | Reviewable changes, passing required checks, full issue evidence, merge-ready rendering branch | #147 and the stacked fixes #154–#159 remain drafts; no merge performed |

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

1. Verify actual hosted Windows test and build execution. The
   [CI and example correction](evidence/windows-example-ci-v1.md) explicitly
   selects Bash and requires execution-summary artifacts. Its hosted run must
   show Cargo output and successful summary contents before this gap is closed.
2. Finish all-example native linking, real starter/example startup, and clean
   Windows installation. Six palette corrections bring the local link audit
   from 13 to 19 successful examples out of 20. The embedded-view example needs
   a newer Perry API, and both available newer Windows bundles have an
   independent standard-library link failure that still needs resolution.
   All-example compilation must become a required PR check.
3. Complete the wider temporal/geometry, performance, memory, resize, and
   capability corpus. The
   [HD surface correction](evidence/windows-ssgi-surface-v1.md) and two valid
   full Radeon runs are [published with #159](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-ssgi-surface-20260911).
   All nine images pass; reproducibility retains 257 byte-identical artifacts.
   The original 16-frame HD startup limits pass on Vulkan, DX12, and hosted
   Metal. Earlier host-load failures remain invalid timing windows. Hosted
   Metal lacks timestamp queries and cannot qualify GPU timing.
4. Complete API generation, streaming, components, runtime UI, and packaging
   against the full issue requirements above, then prepare the draft stack for
   review and integration. These outcomes include both implementation work and
   acceptance evidence; they do not imply that every subsystem is absent.

Local work continues on the Radeon 760M. RTX-specific, physical constrained
adapter, and other unavailable hardware acceptance remains explicitly open.
