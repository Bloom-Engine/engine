# 003 — Stochastic SSR + temporal accumulation

**Effort:** ~2 days · **Expected gain:** SSR 4× cheaper per frame · **Status:** open

## Problem

The SSR pass (`SSR_SHADER_WGSL` in `renderer.rs`) marches 32 steps per pixel at
half-res. Every metallic fragment (and near-metal dielectric at low roughness)
does the full march. With Sponza's chrome lamp fittings and polished marble,
that's a significant pixel count × 32 steps.

## Proposed approach

Replace the deterministic march with **stochastic SSR + TAA-style temporal
accumulation**:

1. **Cast one ray per pixel per frame** instead of 32. Direction sampled from
   the GGX distribution weighted by roughness (importance sampling). Use a
   low-discrepancy sequence (Hammersley + temporal shuffle).
2. **Write noisy reflection into `ssr_rt` same as today.** Single-frame
   reflection is garbage per pixel but unbiased in expectation.
3. **Temporal accumulation in a history texture** — reproject the previous
   frame's accumulated SSR using motion vectors, blend with this frame's noisy
   sample. Converges to a smooth reflection over 4-8 frames. Exactly the
   pattern TAA uses for color, applied to the SSR ray budget.
4. **Rejection heuristic**: if the surface roughness changed significantly
   between frames (e.g. material transition), reset history for that pixel.
5. **Spatial prefilter**: 3×3 bilateral guided by roughness + depth before
   temporal blend. Smooths the worst noise so temporal doesn't need as many
   frames to converge.

The existing TAA pass already does most of this for color — the SSR history
pass is a smaller version of the same idea.

## References

- Stachowiak — "Stochastic Screen-Space Reflections" (SIGGRAPH 2015 Advances
  in Real-Time Rendering)
- Unreal Engine `SSRT.usf` — stochastic variant in UE5
- Frostbite — "Stochastic Screen-Space Reflections" (Zander 2019)

## Acceptance

- `./main --quality 3 --ssr 1 --fps-only 300` — SSR cost drops from baseline
  to ≤ 25% of current.
- Visual: after 4-8 frames of static camera, the marble column reflection
  looks indistinguishable from the 32-step SSR. On rapid camera motion, some
  noise is acceptable (TAA's role to clean up).
- SSIM ≥ 0.98 on static frames.

## Notes for the implementer

- Ray budget: literally `let n_steps = 1u` with jitter; but keep the variable
  so future versions can ramp up on higher-end GPUs.
- Needs a history RT: same format as `ssr_rt`. Ping-pong or two-view pattern.
- The current TAA in Bloom is `taa_pass`. The SSR temporal blend can be a
  separate smaller pass (just SSR reprojection + accumulate) or fold into the
  existing TAA by writing both current + history.
- Preserve the `ssr_enabled` toggle and the quality-preset plumbing.

## Files likely to change

- `native/shared/src/renderer.rs` — rewrite SSR_SHADER_WGSL, add history RT,
  add (or extend existing) reprojection pass.
