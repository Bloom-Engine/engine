# 016 — Lumen importance sampling + hierarchical refinement

**Effort:** 3-5 days · **Expected gain:** quality/ray doubled · **Status:** open

Phase 5 of the [Lumen roadmap](lumen-roadmap.md). Depends on 013. Numbering
skips 015 because the original Phase 4 (HW ray-tracing) was absorbed into
Phase 1 (ticket 007b).

## Problem

007a/007b shoot 32 rays/probe/frame in a **uniform cosine-weighted**
distribution. This wastes ray budget on directions that statistically never
find light (corners, occluded ceilings) while under-sampling the directions
where last frame's strongest contribution came from. Lumen's quality lever is
importance sampling: use prev-frame probe radiance × BRDF as a PDF for the
current frame's ray directions.

A second lever: detect screen tiles with high radiance variance across their
2×2 probe neighbourhood and spawn **extra probes** there at finer-than-16-pixel
stride. This is Lumen's "hierarchical refinement".

## Approach

### Importance sampling

Add a per-probe **structured importance map** (SIM): the prev-frame octahedral
radiance for that probe, treated as a 2D PDF. At ray dispatch:

- For each of the 32 rays/probe, compute a 2D stratified sample (Halton-23
  with blue-noise offset).
- Map the sample through the inverse CDF of the SIM to a direction on the
  sphere. This concentrates rays in high-luminance directions.
- Track per-ray PDF weight and divide at accumulation so the estimator stays
  unbiased.

Implementation: precompute the row-sum + row CDFs of the SIM into a small
auxiliary texture per probe (2× the atlas size). Inverse-CDF sample in WGSL
via two `textureLoad` calls. Standard technique from Physically Based Rendering.

### Hierarchical refinement

After the 007a `probe_filter_pass`, add a `probe_variance_pass`:

- Per 16×16 tile: compute the radiance variance across its 2×2 probe
  neighbourhood + temporal history.
- Tiles with variance above a threshold flag "needs-refinement".

Next frame, dispatch an **extra probe layer** at 8-pixel stride for flagged
tiles only. Store these in a second probe buffer with the same 3D-texture
shape but smaller (only refinement tiles). The resolve pass samples both
layers, preferring the fine layer where present.

Ray budget: uniform 32/probe across the base layer, plus ~8/probe on the
refinement layer for flagged tiles. Net: ~10-15% more rays for substantially
cleaner contact-shadow indirect.

## Files likely to change

- `native/shared/src/renderer/shaders.rs` — importance CDF precompute shader,
  trace-shader updates (both SW and HW variants) to consume the SIM,
  `PROBE_VARIANCE_WGSL`, refinement-layer variant of the trace pass.
- `native/shared/src/renderer/mod.rs` — SIM textures, variance output buffer,
  refinement tile list (indirect dispatch), extra pass wiring.

## Acceptance

- Equal-quality-at-lower-rays: 16 rays/probe/frame with importance sampling
  matches 32 rays/probe/frame without, measured by temporal convergence rate
  on a standing-still camera.
- Equal-rays-higher-quality: 32 rays/probe with importance sampling has
  visibly cleaner noise in shadowed indirect regions (under-arch, column
  interiors).
- Hierarchical refinement: noise along contact-shadow edges (pillar bases,
  under arches) visibly cleaner with refinement on. Tile-flag overhead
  < 0.3 ms/frame.
- FPS: no regression from 007a/007b measurements at `--quality 3 --ssgi 1`.
- Disable-path works: `BLOOM_DISABLE_IMPORTANCE=1` falls back to 007a-style
  uniform sampling cleanly.

## Notes

- Don't over-engineer the refinement tile list. A simple append buffer + counter
  with indirect dispatch is enough for Sponza's tile count (~1 450 base, ~100
  refinement on average).
- Importance sampling also helps HW rays even though HW's per-ray cost is
  similar to uniform — it redirects budget, doesn't save per-ray time.
- This is the *last* Lumen ticket. Beyond this, infinite bounces and SSRC's
  "probe occlusion cone" are natural follow-ups but out of scope here.
