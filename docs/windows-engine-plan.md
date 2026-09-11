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
| #140 integration gates | Same required local/hosted lanes pass on exact source; release package startup and all-example evidence | #160 fixes silent Windows CI non-execution and MSVC PATH ordering. #161 passes the actual native engine build and all 20 native links locally and in hosted CI. #162 fixes the focused DX12 failures; #163 fixes camera-history reset. #164 passes all 22 hosted Tests jobs using an explicit FXC Windows lane. The underlying WARP/DXIL crash remains open. A separate layered-material correction passes the full local FXC shared suite and all 93 DXC/Vulkan goldens. Fresh installed headless scene/direct-2D rendering and cleanup pass through #169. Visible presentation and release packaging remain open. #170's initial Windows shared job again crashes despite FXC; the serial follow-up and #171 each pass all 22 hosted Tests jobs. The driver root cause remains open |
| #138 capability fallback | Actual constrained-adapter startup and relevant forced-tier corpus, truthful capability outputs | Existing implementation/evidence preserved; physical constrained-limit acceptance still needs proof |
| PR integration | Reviewable changes, passing required checks, full issue evidence, merge-ready rendering branch | #147 and the stacked fixes #154–#172 remain drafts; no merge performed |

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

1. **Complete the example runtime experience (#142/#74).**
   The installed one-command starter is qualified through #173 at `955ac5b`:
   all 22 Tests jobs pass, including real asset/text/square browser rendering,
   eight frames and one cleanup. #174's six native game-content/cleanup checks
   and all 20 native links pass locally. Hosted run 34580890147 qualifies five
   browser games; Voxel Sandbox stops on a rejected pointer-lock request. The
   native public-constant fixture exposes colliding module names across `C:`
   and `D:`. The follow-up handles pointer-lock denial, creates linked compiler
   projects beside the checkout and rejects duplicate generated globals. Its
   local native/WASM constants, six native games, six browser builds and all
   20 native links pass; the new hosted run is still required. Full gameplay,
   input interaction and the broader canonical runtime matrix remain open.
   A separate local grid-line candidate restores visibility on DX12/Vulkan,
   but ten existing reconstruction/quality tests fail. It remains incomplete.
2. **Finish and qualify fixed lifecycle integration.**
   The [fixed lifecycle candidate](evidence/fixed-game-lifecycle-v1.md) passes pure
   native/WASM timing and hook-order contracts, plus exact installed rendering
   and cleanup on Radeon DX12/Vulkan. Its revised installed starter passes native
   build/render/asset/cleanup and the full web build locally.
   [#172's published evidence](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-fixed-lifecycle-20260911)
   now includes all 22 passing hosted Tests jobs, the pure native/WASM contract,
   three exact installed native images and the eight-frame compiled browser
   lifecycle with one cleanup. The native package report preserves its dirty
   checkout flag and seven verified installed runtime hashes. Later full starter
   execution found a named-void callback failure addressed by #173; #172's web
   build alone does not establish successful full starter browser execution.
   Pause/focus policy and device-loss recovery remain separate.
3. **Continue Windows integration and packaging (#140/#145).**
   #170, #171 and #172 pass all 22 hosted Tests jobs with a serial Windows harness.
   The original access violations remain retained and their root cause unresolved.
   Installed headless scene/direct-2D modes render exact frames, simulate Jolt and
   clean up once. Visible presentation, packaged DXC/DXIL, clean-machine starter
   setup and general Windows long paths remain incomplete. Physical Radeon
   measurements and hosted software rendering remain distinct evidence.
4. **Complete wider graphics and performance acceptance.**
   [Two strict full Radeon runs at #159](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-ssgi-surface-20260911)
   pass all nine images and reproduce 257 artifacts byte-identically. Rerun
   current-source qualification after integration, and complete representative
   temporal/geometry scenes, fractional/native and frozen A/B timing, memory,
   resize and constrained-adapter checks. Hosted Metal without timestamp queries
   cannot qualify GPU timing. Named discrete hardware acceptance stays open.
5. **Finish the remaining engine systems in dependency order.** Establish the
   safe generated API/UTF-8/ownership contracts (#141), then bounded asynchronous
   asset streaming and cancellation (#137). Use those contracts to finish
   component/prefab lifecycle and destruction safety (#143). Complete runtime UI
   input/layout/accessibility (#144) and platform packaging (#145), then prepare
   the draft stack for review and integration. The full issue requirements in
   the table above govern completion; each subsystem already has some code.

The [current example-loop candidate](evidence/example-loops-v1.md) migrates
test3d, dungeon-crawl, isometric-rpg, kart-racer, space-blaster and voxel-sandbox
onto the shared loop and cleanup. Native execution exposed fractional math,
voxel indexing and camera/derived-coordinate failures. Their corrections pass
the public scalar contract in native/WASM and six actual native render/content/
cleanup checks. Earlier missing-content images are rejected by the new gate.
The all-20 native compile/link gate remains required. Hosted candidate acceptance,
actual example browser runtime, gameplay/input and the missing test3d grid lines
remain open; a nonblank startup image does not complete those requirements.

The [public constants and browser runtime follow-up](evidence/example-browser-v1.md)
finds that WASM barrel re-exports lose palette and input bindings. Explicit
initialized exports retain shared values through `bloom/core` and the package
root, with actual native/WASM constant and alias contracts. All six compiled
games then pass recording-FFI argument checks. Real browser content/cleanup
acceptance is now required in CI. #174's initial hosted native checks all fail
before capture with a numeric-argument TypeError; that failure is preserved and
the shared-binding correction still requires hosted qualification.

At `da31009`, hosted run 34580890147 qualifies five actual browser games;
voxel-sandbox stops on a rejected pointer-lock request. Its native constant
fixture exposes merged module globals when a temporary project on `C:` links
the checkout on `D:`. The follow-up keeps linked test projects beside the
checkout, rejects duplicate generated globals, and handles pointer-lock denial
without stopping rendering. Those corrections still need hosted acceptance.

Local work continues on the Radeon 760M. An RTX 4080 is not a prerequisite for
this implementation work. RTX-specific and physical constrained-adapter evidence
remain explicitly unperformed. No draft PR has been merged or npm package released.
