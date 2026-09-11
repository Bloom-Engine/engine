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
| #140 integration gates | Same required local/hosted lanes pass on exact source; release package startup and all-example evidence | #160 fixes silent Windows CI non-execution and MSVC PATH ordering. #161 passes the actual native engine build and all 20 native links locally and in hosted CI. #162 fixes the focused DX12 failures; #163 fixes camera-history reset. #164 passes all 22 hosted Tests jobs using an explicit FXC Windows lane. The underlying WARP/DXIL crash remains open. A separate layered-material correction passes the full local FXC shared suite and all 93 DXC/Vulkan goldens. Fresh installed headless scene/direct-2D rendering and cleanup pass through #169. Visible presentation and release packaging remain open. #170's initial Windows shared job again crashes despite FXC; serial mitigation awaits hosted qualification |
| #138 capability fallback | Actual constrained-adapter startup and relevant forced-tier corpus, truthful capability outputs | Existing implementation/evidence preserved; physical constrained-limit acceptance still needs proof |
| PR integration | Reviewable changes, passing required checks, full issue evidence, merge-ready rendering branch | #147 and the stacked fixes #154–#170 remain drafts; no merge performed |

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

1. **Finish real compiled-game browser acceptance (#74/#142).**
   The [compiled-game gate](evidence/compiled-web-startup-v1.md) catches a gap
   in the earlier JavaScript-driven renderer check: Perry returned success for
   unresolved imports and emitted a game without the engine calls. The corrected
   preparer installs the exact checkout as a dependency, rejects unresolved
   imports and inspects the actual WASM import table. A local recording-FFI probe
   passes callback/cleanup and explicit startup-fault controls; hosted rendering
   must still produce the exact frame. Perry's plain throw propagation remains
   a separate limitation found during this work.
2. **Qualify the Windows CI mitigation (#140).**
   All 22 Tests jobs passed through #169. #170's initial shared-library job then
   hit an access violation despite FXC. The next attempt serializes the Windows
   harness while retaining every assertion. The local serial library passes
   489 tests with one existing ignored test; that does not establish the crash's
   cause or qualify every helper on WARP. Physical Radeon DX12/DXC and Vulkan
   image evidence remains distinct from hosted software rendering.
3. **Complete the starter and example experience (#142/#145).**
   The installed web command works on Windows. Fresh native packages render
   exact scene/direct-2D frames and simulate Jolt locally and in hosted CI.
   [Shared cleanup, corrected example palettes and Pong pause replay](evidence/windows-game-cleanup-v1.md)
   are [published at #169](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-starter-lifecycle-20260911).
   All 20 canonical native examples link. The [starter command](starter.md)
   creates a project from an installed package; its unmodified native/web builds
   and bounded native greeting/asset/cleanup run pass locally. Default creation
   now carries the exact engine archive, with hosted packaging checks pending.
   All-example web/runtime acceptance, fixed updates, visible native presentation,
   packaged DXC/DXIL and general Windows long-path support remain incomplete.
4. **Complete wider graphics and performance acceptance.**
   [Two strict full Radeon runs at #159](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-ssgi-surface-20260911)
   pass all nine images and reproduce 257 artifacts byte-identically. Rerun
   current-source qualification after integration, and complete representative
   temporal/geometry scenes, fractional/native and frozen A/B timing, memory,
   resize and constrained-adapter checks. Hosted Metal without timestamp queries
   cannot qualify GPU timing. Named discrete hardware acceptance stays open.
5. **Finish API generation, streaming, components, runtime UI and packaging**, then
   prepare the draft stack for review and integration. The full issue requirements
   in the table above govern completion; each subsystem already has some code.

Local work continues on the Radeon 760M. An RTX 4080 is not a prerequisite for
this implementation work. RTX-specific and physical constrained-adapter evidence
remain explicitly unperformed. No draft PR has been merged or npm package released.
