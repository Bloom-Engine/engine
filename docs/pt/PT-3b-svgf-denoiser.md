# PT-3b — SVGF denoiser for the realtime mode

Status: **landed & verified** (2026-07-14, 760M / DX12+DXC).

The user-facing complaint this fixes: realtime mode looked like "a dirty
lens with small ice crystals on it" — a static, screen-glued grain
pattern. The round-7 frozen sample sequence produced exactly that by
design (a wrong-but-stable 1-spp estimate, spatially filtered), and it
turned out to be masking two load-bearing bugs that had disabled ALL
temporal accumulation since PT-3 M1.

## What the realtime mode is now (canonical SVGF, Schied et al. 2017)

- **Rolling RNG in every mode.** The frozen sequence is gone; unbiased
  fresh samples each frame are what the temporal estimator needs.
- **Temporal accumulation with moments.** New moments ping-pong buffers
  (bindings 18/19) carry `(mu1, mu2, history length, raw depth)` per
  trace texel. Color and moments accumulate with `alpha =
  max(1/N, 0.1)` (cumulative average while young, EMA once mature; 0.1
  rather than the paper's 0.2 because the half-res 2-bounce sky lottery
  is the dominant noise source and the deeper average halves the
  residual mottle — direct sun is cascade-deterministic, so the slower
  EMA only delays indirect changes).
- **Bilinear 2×2 reprojection with per-tap depth validation** (tight
  relative tolerance, surface identity) replacing point sampling + the
  old loose-tolerance workaround.
- **Surface-flip write-through.** At half res a trace texel's owner
  pixel alternates between surfaces with TAA jitter (grass blade vs
  ground). When no tap matches but the stored surface is still inside
  the texel's current footprint depth window, the history is kept
  VERBATIM and the sample dropped — one texel holds one surface's
  history; the depth-guided upsampler routes texels to full-res pixels
  by surface. Blending would leak blade lighting onto the ground
  (gray-blue mottle); resetting would pin the texel at 1 spp (speckle).
  Only a stored surface that left the footprint is a disocclusion.
- **Variance-guided à-trous** — five iterations (steps 1/2/4/8/16), B3
  5×5 kernel, `sigma_l = 4·sqrt(gaussian3x3(variance))`, variance
  filtered alongside with squared weights, depth-gradient edge stop.
  Young history (< 4 frames) substitutes a spatial variance estimate
  and floors it at 0.25 so disoccluded texels blur instead of surviving
  as false edges.
- **First-iteration feedback:** iteration 1's output is copied back
  over the accum buffer as next frame's color history (moments stay
  raw) — the SVGF detail that makes the loop stable at 1 spp.
- Removed: history spike clamp, despeckle max-clamp, fixed sigma
  schedule, frozen IGN. Kept (documented deviation): a single
  irradiance-space firefly clamp at 4.0, as reference implementations
  do.

## The two masked bugs (both invisible under the frozen seed)

1. **`tlas_version` reset nuked realtime history every frame.**
   `pt_accum_count` was zeroed whenever the TLAS version changed — and
   it changes on every node transform, i.e. every gameplay frame.
   Correct scoping: full reset for progressive only; realtime's
   per-texel depth validation already invalidates exactly what changed.
   Found via the debug-20 history-length view (black everywhere).

2. **`prev_vp` was uploaded transposed — reprojection was garbage under
   any camera motion.** The engine has TWO matrix storage conventions:
   `mat4_invert` outputs land transposed relative to WGSL `M*v` (hence
   inv_vp's transpose-at-upload), while `mat4_multiply` products are
   already in `M*v` layout (the shadow cascade VPs upload raw for the
   same reason). Transposing the composed prev-VP collapsed every
   reprojection into a ~40-texel band at screen centre with a
   nonsensical depth. Found via the debug-23 numeric dump; after the
   fix the dump shows sub-texel exact reprojection (pos tracks column
   1:1, depth pair agrees to 0.1%) with the title camera orbiting.

## Verification

- Debug views: 20 = history length heat, 21 = variance ×10,
  22 = numeric dump (n / variance / tap mass / depth),
  23 = numeric dump of reprojected position + depth pair.
  All dump through `pt_trace_dump.txt` (BLOOM_PT_DEBUG env).
- Ground truth: converged PROG stills (25 s static) — smooth dark
  ground under grass; RT now matches its structure (the earlier
  gray-blue mottle was cross-surface leakage, not signal).
- Shimmer meter (two stills 1.5 s apart, grass band):
  **RT 37.6 mean / 15.9 % sparkle vs raster 49.5 / 16.7 %** — the path
  traced mode is temporally QUIETER than the raster baseline.
- Cost: ~15 fps RT vs ~17 before at 0.75 render scale (moments buffers
  + two extra wavelet iterations + feedback copy).

## Notes / open

- Known hybrid-sun tradeoff: cascade shadows cannot resolve grass
  micro-shadows, so RT ground under grass is brighter and flatter than
  the traced-sun PROG reference. Deliberate (stability over contact
  detail); traced-sun-in-RT would be a quality toggle, not a fix.
- The write-through's footprint window compares current-frame linear
  depth against a stored prev-clip depth; under fast motion it degrades
  toward disocclusion-reset, which is the safe direction.
