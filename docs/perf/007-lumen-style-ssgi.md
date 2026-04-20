# 007 — Probe-based SSGI (replace per-pixel march)

**Effort:** ~1 week · **Expected gain:** SSGI 5× cheaper · **Status:** open

## Problem

Bloom's current SSGI pass does per-pixel ray marching for indirect diffuse
lighting: 8 samples × 14 march steps = **112 iterations per half-res pixel**,
each with a texture fetch. On Sponza's 800×450 half-res RT that's 40 M
iterations per frame, each with a matrix multiply and a texture fetch.

UE5 Lumen replaces this with **probes**: adaptive screen-space probes at a
~16×16-pixel granularity, each probe gathers its own GI, then the full-res
pass interpolates probe radiance guided by depth/normal. Per-pixel work
collapses to a few texture taps.

## Proposed approach

A screen-space probe system (simpler than full Lumen — Lumen also uses SDF
tracing which needs GPU SDFs of the mesh, out of scope here):

1. **Probe placement**: a grid of probes on a 16-pixel stride at half-res (so
   one probe per 16×16 screen block = 50×30 = 1500 probes for a 800×450 RT).
2. **Probe march**: each probe ray-marches from its world position in N
   directions (e.g. 16 directions per probe), using the existing depth +
   radiance buffers. Total rays: 1500 × 16 = 24k per frame. Compare to
   current 360k × 8 = 2.9 M per frame — 120× fewer rays.
3. **Temporal accumulation**: each frame, a probe casts fewer rays (say 4)
   and accumulates into a per-probe history. After 4 frames, a probe has
   sampled all 16 directions.
4. **Per-pixel reconstruction**: full-res pixel samples the 4 nearest probes
   and reconstructs irradiance via bilinear + depth/normal weighting (same
   idea as irradiance probes in offline GI).
5. **Adaptive density**: in low-frequency regions (uniform walls), drop to
   one probe per 32 pixels. In high-frequency (near edges/corners), keep
   16-pixel density. Start without adaptive — measure flat first.

The existing `ssgi_temporal_pass` (ping-pong history) can be repurposed as
the probe history texture; architecturally the shape is similar.

## References

- Lumen Technical Details — "Global Illumination and Reflections in
  Unreal Engine 5" (SIGGRAPH 2022 Advances)
- Majercik et al — "Dynamic Diffuse Global Illumination with Ray-Traced
  Irradiance Fields" (JCGT 2019) — DDGI, the probe interpolation approach
- Epic's Lumen screen-space probe source in UE5
  (`LumenScreenProbeGather.cpp`)

## Acceptance

- `./main --quality 3 --ssgi 1 --fps-only 300` ≥ 2× FPS improvement from
  current SSGI-enabled baseline.
- Screenshot: indirect bounce colour on shaded side of column / under arches
  looks similar to baseline. Some low-frequency noise on camera pan is
  acceptable (temporal accum converges over frames).
- Edge cases: rapid camera movement should not cause probe ghosting — use
  disocclusion rejection on the probe history.

## Notes for the implementer

- Probes live in a small structured buffer (per-probe position + direction
  weights + radiance accumulator). Size: 1500 probes × 16 dirs × 4 bytes =
  ~100 KB.
- Probe generation can run on the graphics queue before the full-res
  reconstruction — or async compute for ticket 010 synergy.
- Falls back gracefully: when SSGI is disabled, the composite samples a
  zero-filled probe texture and contributes nothing.

## Files likely to change

- `native/shared/src/renderer.rs` — SSGI_SHADER_WGSL rewrite, probe buffer,
  new reconstruction pass replacing the current `ssgi_pass`.
- `native/shared/src/renderer.rs` (ssgi_temporal_pass) — repurpose as probe
  history or delete.
