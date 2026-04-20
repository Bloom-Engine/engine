# 001 — TSR-style half-res rendering + temporal reconstruction

**Effort:** ~1 week · **Expected gain:** 2× on main_hdr + all post-FX · **Status:** open

## Problem

Every fragment-shader pass in Bloom runs at the full physical surface
resolution. On a Retina Mac that's 1600×900 = 1.44 M fragments per pass. For
Sponza the render chain is:

```
main_hdr (4 MRTs × 1.44 M × PBR shader) ≈ 17 ms
shadow (3 × 2048²)                       ≈ 14 ms
SSAO + blur (half-res 800×450)           ≈ 186 ms  ← biggest
SSR (half-res, 32 march steps)           ≈ varies
SSGI (half-res, 8 × 14 march)            ≈ varies
bloom chain (9 passes)                   ≈ a few ms
TAA, DoF, MB, SSS, composite             ≈ a few ms each
```

Fragment count dominates everything. Halving both axes quarters fragment work.

## Proposed approach

Render the whole chain (main_hdr + every post-FX that's not already half-res)
at **half resolution** (800×450 on a Retina surface). Reconstruct to full
resolution in the final composite step using **temporal upsampling** — the
technique Unreal's TSR, NVIDIA DLSS, and AMD FSR 2 all use.

Key pieces:

1. **Jitter the projection matrix** by a sub-pixel sequence (Halton(2,3) or
   8-tap R2 sequence) every frame. Scale the jitter to the *half-res* pixel.
2. **Render main_hdr + SSAO + SSR + SSGI + bloom etc. at half-res.** All RTs
   drop to `(surf_w / 2, surf_h / 2)` instead of surface size.
3. **Build a history buffer at full resolution.** Each frame the reconstruction
   pass samples the current half-res color + the previous-frame full-res
   history with reprojection (motion vectors from the velocity RT we already
   write).
4. **Neighborhood clamp + variance estimation** to kill ghosting — sample the
   3×3 neighborhood of the current half-res color, compute min/max per channel,
   clamp history to that AABB in YCoCg or similar perceptual space.
5. **Disocclusion heuristic** — if the reprojected history sample came from
   behind current geometry (depth test), accept current frame only (no blend).

The existing TAA pass can be upgraded in-place — it already does reprojection,
just needs to become an *upscaling* reprojection that reads half-res color and
writes full-res history.

## References

- Karis — "High Quality Temporal Supersampling" (GDC 2014) — original TAAU/TSR
- Unreal Engine 5 TSR source (Engine/Shaders/Private/TemporalSuperResolution/)
- Epic's GDC 2022 talk on UE5 TSR
- "Temporally Stable Real-Time Joint Neural Denoising and Supersampling"
  (Intel, 2021) for the motion-vector dilation trick

## Acceptance

- `./main --fps-only 300` ≥ 30 fps (target: 60 fps combined with ticket 002).
- `./main --capture 30 /tmp/after.png` vs `/tmp/sponza_baseline.png` must not
  show visible aliasing, ghosting on camera pan, or detail loss on stone
  texture / shadow edges. Compare with `compare -metric SSIM` (target SSIM
  ≥ 0.98 on the full image).
- `./main --quality 0 --fps-only 60` still at 60 fps.

## Notes for the implementer

- The `--profile N` harness records `render_total` and per-pass CPU. Use
  `--fps-only` for real timing — GPU timestamps on Metal are unreliable for
  small passes.
- The velocity RT is already written by `main_hdr_pass` — use it for
  reprojection without plumbing new data.
- The current TAA pass uses a neighborhood clamp already; study its shader
  (`taa_pass` in `renderer.rs`) before replacing.
- Jitter sequence should reset on camera cut (call sites: `begin_mode_3d`).
- Consider disabling the reconstruction when the user passes
  `--quality 0` so the "bare" test case stays comparable.

## Files likely to change

- `native/shared/src/renderer.rs` — all RT sizes, TAA pass rewrite, new
  reconstruction pass, jitter calculation
- `native/shared/src/renderer.rs` (SSAO/SSR/SSGI/bloom RT creation helpers)
