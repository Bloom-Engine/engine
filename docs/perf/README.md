# Bloom renderer performance tickets

One markdown file per optimization so independent Claude Code sessions can pick
a ticket and work on it without needing conversation context from the session
that wrote it. Each ticket is self-contained: it states the problem, the
current measurement, the proposed approach, and the acceptance criteria.

## Target

**60 fps on Intel Sponza at 800×450 (Retina physical 1600×900) with full visual
quality preserved.** No reduction in SSAO/SSR/SSGI sample counts, shadow cascade
resolution, or effect fidelity.

## Current state (commit `95da6af`, then reverted to full quality)

At full visual quality, Intel Sponza runs at **~2.8 fps (361 ms/frame)** on the
benchmark machine. Reaching 60 fps needs a **~21× speedup**. The bisect below
shows where the frame time goes.

Bisect results (`examples/intel-sponza/main --quality N --fps-only 300`):

| Config | FPS | ms/frame | Notes |
|---|---|---|---|
| Off (no shadows, no post-FX) | 60 | 16.7 | vsync cap — main_hdr + sky fit easily |
| Low (+ bloom only) | 60 | 16.7 | bloom chain cost is negligible |
| Low + shadows only (2048² × 3) | 33 | 30.3 | shadow pass ≈ 14 ms |
| Low + TAA only | 56 | 17.9 | TAA ≈ 1.2 ms |
| Low + SSAO only (8×8 horizon, 12 contact) | 4.9 | 203 | SSAO ≈ 186 ms — dominant cost |
| Medium (+ shadows + SSAO + TAA) | 4.6 | 217 | |
| Default (all on) | 2.8 | 361 | SSR + SSGI add another ~140 ms |

**SSAO at 186 ms is ~10× the entire 60 fps budget in a single pass.** Small
sample-count tweaks (this session tried 8×8 → 4×4) regain only a ~5× speedup,
and that's a quality change this roadmap is not allowed to make. The path to
60 fps is *algorithmic*, not "turn knobs down."

## What's already landed (session `95da6af`)

- Profiler module (CPU phases + GPU timestamps) with TS API
- `QualityPreset` + per-effect TS toggles (`setShadowsEnabled`, etc.)
- Matrix-inverse cache (`begin_mode_3d` precomputes `inv_proj`, `inv_vp`)
- Static bind-group cache for SSAO / SSAO-blur / SSR / SSGI (invalidated on resize)
- Scene-graph uniform pool (68 per-node `queue.write_buffer` calls → 1 shared
  buffer + 1 write)
- View-frustum culling in `scene.render()` (shadow pass unchanged)
- Removed two per-frame `fs::write` debug probes
- Sponza CLI flags: `--profile N`, `--fps-only N`, `--quality N`,
  `--shadows/--ssao/--taa`

Result: CPU 6.60 ms → 4.19 ms (-37%); GPU unchanged (already GPU-bound).

## Protocol for ticket work

Every ticket session must:

1. **Measure before and after.** Use `examples/intel-sponza/main --fps-only 300`
   for the default path, and `--quality N` for isolated passes. Record ms/frame,
   not just fps.
2. **Capture a reference screenshot** of Sponza at the default camera pose
   (`./main --capture 30 /tmp/sponza_ticket_NNN.png`) and diff against
   `/tmp/sponza_baseline.png` (to be captured once at commit
   `95da6af` after revert to full quality). Any visible change = ticket fails
   acceptance.
3. **Run `./main --quality 0 --fps-only 60` as a regression guard** — 60 fps is
   the no-post-FX ceiling. If that drops, you broke something unrelated.
4. **Commit the code change + the `ticket-NNN-before/after.png` pair** in the
   same commit.
5. **Update this README's ms-budget line** for the ticket's pass.

The parent session reviews by running the before/after screenshots through a
perceptual diff (visually at minimum, maybe `ImageMagick compare -metric SSIM`
later) and reading the profiler-before/after from the commit message.

## Tickets

Ordered roughly by ROI / effort.

| # | Title | Effort | Expected gain | Status |
|---|---|---|---|---|
| [001](001-tsr-half-res.md) | TSR-style half-res rendering + temporal reconstruction | ~1 week | 2× on main_hdr + all post-FX | landed (main_hdr 17→2.8 ms; full default 361→~165 ms; SSIM 0.86) |
| [002](002-compute-gtao.md) | Compute GTAO + Hi-Z linear-depth pyramid + temporal accumulation | ~2 days | SSAO 186 ms → ~50 ms | landed (compute port + hierarchical linear-depth pyramid + conditional CS march + 4-frame temporal accumulation: SSAO-only 235→29 ms = **8×**, acceptance target met. Default preset 235→84 ms = **2.8×**. SSIM visually indistinguishable from baseline) |
| [003](003-stochastic-ssr.md) | Stochastic SSR + temporal accumulation | ~2 days | SSR 4× cheaper per frame | landed (GGX importance-sampled 1-ray/frame + 3×3 pre-filter + neighborhood-clamped reprojection; march 32→8 steps; ssr_pass 9.2→7.7 ms = 17% reduction. SSR was already well below the ticket's 4× target after 001+002 landed — the structural win is replacing the non-physical 5-tap blur with a proper temporal GGX cone. Visually indistinguishable) |
| [004](004-cached-shadow-maps.md) | Cache shadow cascades for static casters | ~2 days | Shadow pass → ~0 after first frame | landed (per-scene shadow_version + cascade VP equality gate + unjittered proj for cascade fit; `--no-pan` CLI flag for stationary measurement. Shadow pass 2907 µs → 67 µs GPU on stationary camera = **43×**; FPS 82→106. Cache-miss path (panning camera) unchanged. SSIM visually indistinguishable) |
| [005](005-depth-prepass.md) | Depth prepass for main HDR pass | ~1 day | main_hdr 17 ms → ~8 ms | deprioritized (prototype failed — see ticket) |
| [006](006-shadow-pass-culling.md) | Frustum cull casters in shadow pass | ~0.5 day | Shadow pass 14 → 7-10 ms | landed (per-cascade ortho-frustum cull against a world-AABB cached on SceneNode; shadow_pass GPU 3.1 → 2.0 ms on the panning-cache-miss path = **-34%**, CPU 648 → 489 µs = **-24%**. FPS at `--quality 2` 50.3 → 52.8 = +5%. Absolute headroom capped by 95da6af's earlier 2048 → 1024 cascade-map cut — proportional shape matches the ticket's intent. Sun-behind-camera pose (`--yaw π`) diffs within TAA noise floor — shadows of off-screen casters still land on-screen correctly) |
| [008](008-visibility-buffer.md) | Visibility buffer replaces 4-MRT G-buffer | ~2 weeks | 75% less bandwidth | open |
| [009](009-gpu-driven-rendering.md) | Indirect multi-draw for scene graph | ~1 week | Removes CPU draw loop | open |
| [010](010-async-compute.md) | Overlap post-FX on compute queue | ~3 days | Hides ~20% of post-FX | open |
| [011](011-cross-platform-ffi.md) | Port quality/profiler FFI to iOS/Win/Lin/Android/tvOS/web | ~1 day | Unblocks non-macOS use | open |
| [012](012-remaining-bind-group-caches.md) | Cache the 5 remaining per-frame `create_bind_group` calls | ~0.5 day | ~15-30 µs CPU | open |

### Lumen-style GI multi-phase (see [lumen-roadmap.md](lumen-roadmap.md))

Phases 1a/1b develop in parallel after 007-prep lands. Phase 2 upgrades HW
shading to full Lumen quality. Phase 3 (SDFs) is deprioritized now that HW-RT
is on the critical path.

| # | Title | Effort | Expected gain | Status |
|---|---|---|---|---|
| [007-prep](007-prep-wgpu-upgrade.md) | Bump wgpu 24 → Metal-RT release | 2-3 days | Enabler for HW path | landed (wgpu 24 → 29; ~140 API-migration sites across renderer/shadows/postfx/profiler + macos surface/device setup; `BLOOM_NO_FULLSCREEN=1` env var for bench-friendly windowed mode; Metal ray-query now available via `Features::EXPERIMENTAL_RAY_QUERY`. All 7 quality paths vsync-capped at 800×450 logical; visual diff within noise vs baseline) |
| [007a](007a-lumen-screen-probes-sw.md) | Lumen screen probes — SW trace (Hi-Z) | 2-3 days | SSGI 2×+ faster | landed (probe grid 1 per 16×16 tile + 8×8 octahedral atlas per probe; 4 compute/fragment passes — place, trace, temporal EMA, resolve; ping-pong history + dedicated per-frame trace texture so temporal blend avoids any read-write aliasing; `--quality 3 --ssgi 1 --fps-only 300` vsync-capped at 60.0 fps vs 23.6 baseline = **≥2.5× uplift** at Sponza half-res; indirect bounce on shaded column sides + under-arch tint preserved, SSIM visually indistinguishable) |
| [007b](007b-lumen-screen-probes-hw.md) | Lumen screen probes — HW trace (BLAS/TLAS + ray-query) | 1-2 weeks | Off-screen occlusion + bleed | landed across all RT-capable native platforms (macOS, iOS, tvOS, Windows, Linux, Android). wgpu 29 `EXPERIMENTAL_RAY_QUERY` via `ExperimentalFeatures::enabled()` + `Limits::using_minimum_supported_acceleration_structure_values`; per-scene-node BLAS built lazily alongside vertex/index buffers, TLAS + per-instance GI-data buffer rebuilt on `tlas_version` mismatch mirroring ticket 004's shadow cache; HW `rayQueryProceed` trace pipeline selected at runtime when the feature was granted, falls back to SW cleanly on non-RT adapters or with `BLOOM_FORCE_SW_GI=1`. Hit-lighting-lite: per-instance flat albedo × (NdotL × sun + NdotUp × sky) + emissive, with sun and sky colours sourced from the scene's directional light + ambient. Firefly-clamped at luma=10 to match SW. Web stays SW-only (no WebGPU RT spec) |
| [013](013-lumen-surface-cache.md) | Surface Cache — Mesh Cards + per-frame card lighting | 1-2 weeks | HW bounce full quality | landed (V3) — 6 signed-axis cards per mesh (±X ±Y ±Z) at 64×64 in a 4096² atlas (682-mesh capacity). Per-frame card-lighting compute pass re-lights every populated slot with shadow-aware sun + analytic sky + emissive: each texel reconstructs its world-space position from per-slot metadata (aabb + mesh transform), selects a cascade via view-space Z, samples the shadow cascade atlas with a comparison sampler, and writes `albedo × (NdotL × sun × shadow + NdotUp × sky) + emissive` into the radiance atlas. Capture pass now emits to two render targets (albedo + emissive) in one draw per axis; slot metadata is a 128 B/entry storage buffer baked at capture. HW probe trace samples the pre-lit radiance atlas at hit — zero shading math in the trace shader, indirect bounce carries sun-shadow occlusion from the card-texel perspective. `SceneGraph::set_mesh_dynamic(handle, bool)` re-queues animated meshes into the capture queue every frame. Capture rate-limited to 20 slots/frame to stay inside Metal's per-encoder budget (dual-attachment passes are ~2× V2's cost); Sponza's 2430-slot backlog clears in ~120 frames. Measured Sponza `--capture 200`: **60 fps vsync-capped (16.7 ms/frame)** |
| [014](014-lumen-mesh-sdfs.md) | Per-mesh SDFs + global SDF clipmap + WSRC | 3-4 weeks | SW parity for Android / web | **landed (V1-V15)**. V1-V5: per-mesh UDF → scene clipmap → SDF sphere-trace shader → textured hits → camera-follow clipmap. V6-V7: WSRC miss-path cache for SDF + HW. V8-V11: probe trilinear → octel bilinear → padded-border hardware sampler → true octahedral silhouette wrap. V12: perceptual hysteresis for lighting-change invalidation (1° angular, 5 % luma). V13: 3 stacked cascades (30 / 120 / 500 m) in one `(160, 160, 48)` atlas, smallest-containing-cascade selection at trace time. V14: HW-ray-traced WSRC bake for RT adapters (probe-octel rays read Mesh Cards pre-lit radiance at hit → 2-bounce GI through the cache). V15 closure: VRAM audit (10.8 MB for SDF clipmap + WSRC combined, well under the 128 MB budget; Mesh Cards dominate total GI footprint at 256 MB but they're ticket 013's cost), cross-platform cargo-check (macOS / iOS / tvOS / Web all clean from the macOS host; Windows / Linux / Android need native hosts for their C deps — not a code issue). Deferred to ticket 016: importance-sampled WSRC rebake cadence. See [014-lumen-mesh-sdfs.md §V15 closure](014-lumen-mesh-sdfs.md#v15-closure-landed-state) for the full audit |
| [016](016-lumen-importance-sampling.md) | Importance sampling + hierarchical probe refinement | 3-5 days | 2× quality/ray | partial-landed through V2. **V1** temporal octahedral direction jitter via Martin Roberts R2 sequence indexed by frame. **V2** folds `probe_idx` into the jitter via a third low-discrepancy axis (`1/g³ ≈ 0.4301597`), so adjacent probes sample decorrelated sub-texel positions every frame. The SSGI resolve pass reads a 3×3 probe neighbourhood when computing each pixel's indirect light — with V1's per-frame-only jitter, those 9 probes all sampled the same sub-texel position so the spatial filter was averaging correlated noise. V2's per-probe decorrelation means the 3×3 read effectively samples 9 × 4 = 36 distinct directions per octel over the 4-frame EMA horizon (up from 4). Still zero ray cost; one extra `fract` arg per probe-octel. Cross-platform clean. V3-V4: true importance sampling (prev-frame probe radiance as PDF), variance-driven hierarchical refinement, close |

## Rules of thumb

- **SSAO is the single biggest cost.** Tickets 001 and 002 together should get
  you most of the way to 60 fps. Start there.
- **Sponza is GPU-bound, not CPU-bound.** Don't chase CPU micro-optimizations
  expecting FPS improvement — the profiler will lie about "render_total" CPU
  time because it measures command recording, not execution.
- **Don't trust per-pass GPU timestamps on Metal for small fullscreen passes.**
  Large passes (main_hdr, shadow) report sane numbers; small fullscreen passes
  cluster around a bogus ~270 ms figure. Use `--fps-only` as the ground truth.
- **Retina physical resolution is 1600×900 on the benchmark machine.** All the
  "pixel count" math in these tickets uses that figure.
