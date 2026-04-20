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
| [003](003-stochastic-ssr.md) | Stochastic SSR + temporal accumulation | ~2 days | SSR 4× cheaper per frame | open |
| [004](004-cached-shadow-maps.md) | Cache shadow cascades for static casters | ~2 days | Shadow pass → ~0 after first frame | open |
| [005](005-depth-prepass.md) | Depth prepass for main HDR pass | ~1 day | main_hdr 17 ms → ~8 ms | open |
| [006](006-shadow-pass-culling.md) | Frustum cull casters in shadow pass | ~0.5 day | Shadow pass 14 → 7-10 ms | open |
| [007](007-lumen-style-ssgi.md) | Probe-based SSGI (replace per-pixel march) | ~1 week | SSGI 5× cheaper | open |
| [008](008-visibility-buffer.md) | Visibility buffer replaces 4-MRT G-buffer | ~2 weeks | 75% less bandwidth | open |
| [009](009-gpu-driven-rendering.md) | Indirect multi-draw for scene graph | ~1 week | Removes CPU draw loop | open |
| [010](010-async-compute.md) | Overlap post-FX on compute queue | ~3 days | Hides ~20% of post-FX | open |
| [011](011-cross-platform-ffi.md) | Port quality/profiler FFI to iOS/Win/Lin/Android/tvOS/web | ~1 day | Unblocks non-macOS use | open |
| [012](012-remaining-bind-group-caches.md) | Cache the 5 remaining per-frame `create_bind_group` calls | ~0.5 day | ~15-30 µs CPU | open |

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
