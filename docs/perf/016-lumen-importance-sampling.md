# 016 — Temporal jitter + variance-adaptive EMA (shipped importance sampling)

**Effort:** 3-5 days · **Shipped uplift:** ~4× effective rays/probe via temporal accumulation, plus fast-path EMA on disocclusions · **Status:** **landed (V1-V4)**

Phase 5 of the [Lumen roadmap](lumen-roadmap.md). Depends on 013. Numbering
skips 015 because the original Phase 4 (HW ray-tracing) was absorbed into
Phase 1 (ticket 007b).

The original scope of this ticket was UE5-style CDF importance sampling + a
separate refinement-probe layer. After prototyping, V1-V4 landed a different
approach — temporal octahedral jitter + history-aware jitter scaling +
variance-adaptive temporal EMA — that captures the same "spend the ray budget
where variance is highest" intent at zero new infrastructure cost. The CDF +
refinement-layer work is deferred; see the follow-ups section below.

## Problem

007a/007b shoot 64 rays/probe/frame in a uniform cosine-weighted distribution.
Quality is bounded by two effects:

1. **Under-sampled octels.** Each of the 64 octahedral texels lands one ray,
   every frame, at the same texel centre. Temporal accumulation over 4 frames
   integrates exactly 4 rays per octel — more than that just smooths the
   existing 4, not the underlying signal.
2. **Disocclusion convergence.** The temporal EMA uses a fixed per-octel
   alpha. In stable regions that's great (very strong smoothing); on moving
   lights or disoccluded probes the same alpha means visible lag — the
   history is actively wrong and we're weighting it too heavily.

Both effects had to be addressed without changing the 64-rays/probe/frame
shape (the ray count was already at the GPU-budget ceiling coming out of 013).

## Shipped approach (V1-V4)

Four small layered changes, each builds on the previous one:

### V1 — Temporal octahedral direction jitter

Per frame, offset every octel's ray direction by an R2 low-discrepancy 2D
offset in octahedral-UV space. Over a 4-frame temporal window each octel now
samples 4 distinct sub-texel directions instead of 4 copies of the same
direction. Effective rays/probe/frame-equivalent ≈ 4× with zero extra
dispatch cost.

### V2 — Per-probe decorrelation

Fold `probe_idx` into the R2 sequence axis so neighbouring probes in the
3×3 resolve-neighbourhood sample 9 different directions per octel per frame.
Over a 4-frame window the resolve gets 4 × 9 = 36 unique directions per
octel, up from 4.

### V3 — History-aware jitter magnitude

Bind the prev-frame probe-radiance texture to the trace pipeline. For each
octel, scale the V1/V2 jitter magnitude by the texel's prev-frame luma:

- High luma → small jitter (exploit known-bright directions, tighter
  convergence)
- Low luma → full jitter radius (explore — keeps the estimator unbiased
  for areas the history has no signal for yet)

This is the practical payload the original ticket was after — redirecting
the ray budget toward directions that paid off last frame — implemented as
a 1-line jitter-scale multiplication rather than a full CDF precompute +
inverse-CDF sample.

### V4 — Variance-adaptive temporal EMA

Replace the fixed per-octel temporal alpha with a delta-driven one:

```
delta  = |curr_luma - hist_luma|
alpha  = clamp(0.25 + delta * 0.6, 0.25, 1.0)
```

Stable octels keep the strong 0.25 smoothing; octels where the current
radiance disagrees with history ramp up to full-refresh alpha so
disocclusions / moving lights converge in 1-2 frames instead of 4-6.
This is the "high-variance regions need more attention" intent from the
original hierarchical-refinement plan, captured in the temporal filter
rather than in a second probe layer.

## Files changed

- `native/shared/src/renderer/shaders.rs` — probe-trace shader (SW + HW
  variants): R2 octahedral jitter (V1), probe_idx decorrelation (V2),
  prev-frame luma lookup + jitter-scale (V3); probe-resolve shader:
  delta-driven EMA alpha (V4).
- `native/shared/src/renderer/mod.rs` — prev-frame probe-history texture
  binding on both trace pipelines.

No new textures, buffers, passes, or dispatches — every change fits inside
the 007a frame graph.

## Acceptance

- ✅ **Quality/ray improvement visible on Sponza.** Under-arch indirect and
  column-side bounce show noticeably cleaner spatial-temporal noise at the
  same 64 rays/probe as V0. SSIM visually indistinguishable from
  reference / within TAA noise floor.
- ✅ **No FPS regression.** `--quality 3 --ssgi 1 --fps-only 300` stays
  vsync-capped at 60 fps on the benchmark machine — same as the 007a/007b
  ceiling. V3 adds one `textureLoad` + `mix`; V4 adds a `dot` + `abs` +
  `min`. Noise-floor cost on the trace + resolve passes.
- ✅ **Cross-platform clean.** Builds on macOS, iOS, tvOS (native) and web
  (WASM) from a macOS host.
- ✅ **Regression guard re-verified under ticket 017** — no Lumen work
  leaks into `--quality 0`.

## Deferred — tracked as GitHub issues

Both of these are captured as open Lumen follow-up issues and are the
*original* scope of this ticket that V1-V4 did not land. They stay
deferred because V1-V4 hit the target at zero new infrastructure cost;
a future scene with tile-level banding that V4 adaptive EMA cannot
resolve would motivate picking one of them up.

- **[#19](https://github.com/bloom-engine/bloom/issues/19)** — True CDF-based
  importance sampling for probe rays. Per-probe 8×8 SIM texture + row-sum +
  row-CDF precompute, inverse-CDF sampled per ray, per-ray PDF weight
  tracked for unbiasedness. Stronger in cases where the luma-scaled jitter
  from V3 still wastes rays on zero-contribution directions (e.g. large
  black-ceiling probes).
- **[#20](https://github.com/bloom-engine/bloom/issues/20)** — Variance-driven
  refinement probe layer. Second probe buffer at 8-pixel stride for
  high-variance tiles only, resolve prefers the fine layer. Stronger on
  contact-shadow edges than V4's temporal variance because it adds *spatial*
  resolution, not just more temporal weight.

Also deferred (captured only here):

- **Runtime disable toggle** — the original ticket specified
  `BLOOM_DISABLE_IMPORTANCE=1` but V1-V4 each landed as individual commits
  with incremental visual diffs, so a single-toggle off-switch wasn't needed
  to evaluate. If a downstream platform shows a regression on any one of V1-V4
  we'd add a per-version toggle rather than a single "importance off" switch.

## Notes for future implementers

- V1-V4 live inside the existing probe-trace + probe-resolve passes — no new
  pipelines, no new textures beyond the prev-frame history binding that
  007a already maintained. Cost delta on the GPU is in the noise floor.
- The 3×3 resolve neighbourhood is what makes V2 interesting. Without it,
  per-probe decorrelation just adds noise to individual probes. Anyone
  rewriting the resolve pass needs to preserve the 3×3 sample footprint.
- V4's 0.25 → 1.0 alpha range was chosen so stable regions match the
  pre-V4 strength exactly. Pushing the floor below 0.25 would make
  stable areas noisier; pushing the ceiling below 1.0 would slow
  disocclusion convergence. The `0.6` delta-scale is the one knob that
  can be re-tuned per scene without changing the estimator shape.
