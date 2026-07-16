# 008 — Visibility buffer replaces 4-MRT G-buffer

**Effort:** ~2 weeks · **Expected gain:** 75% less G-buffer bandwidth · **Status:** deferred

## Problem

The `main_hdr_pass` writes four MRTs at the full physical resolution:

| RT | Format | Bytes/pixel |
|---|---|---|
| `hdr_rt`       | Rgba16Float | 8 |
| `material_rt`  | Rg8Unorm    | 2 |
| `velocity_rt`  | Rg16Float   | 4 |
| `albedo_rt`    | Rgba8Unorm  | 4 |

Total: **18 bytes per pixel written** by every fragment. At 1600×900 that's
26 MB per pass per frame, with overdraw multiplying the real write count.
Bandwidth-bound on integrated GPUs.

UE5's Nanite uses a **visibility buffer** instead: store only `(triangle_id,
barycentrics)` (~8 bytes) in the G-buffer, defer material evaluation to the
shading pass. No material sampling happens for hidden pixels. Combined with
depth prepass, every visible pixel shades exactly once.

## Proposed approach

This is a significant refactor — do ticket 005 (depth prepass) first. Then:

1. **Replace main_hdr_pass output** with a single **visibility buffer**: a
   Rgba32Uint texture storing `(triangle_id, u, v, mesh_id)` per pixel.
   Material and normal are *not* written.
2. **New shading pass** reads the visibility buffer, fetches the vertex data
   for the referenced triangle, interpolates attributes from barycentrics,
   and evaluates the full PBR shader once per pixel.
3. **MRTs that post-FX consumes** (normal, albedo, material, velocity) can
   either be rebuilt per-pixel in the shading pass OR kept as separate passes.
   Simplest path: the shading pass writes them alongside the final HDR
   colour — still one write per pixel, vs 4 writes per overdrawn pixel today.

## Simpler intermediate step

If full visibility buffer is too much, consider **dropping unused MRTs when
features are off**:

- `velocity_rt` is only needed when TAA or motion-blur is on.
- `albedo_rt` is only needed when SSGI or SSR is on.
- `material_rt` is needed for SSR and the shadow map sampler stuff.

Rebuild the `scene_pipeline` with 2 or 3 MRT targets when the user has
disabled the dependent post-FX. Cut 30-50% of the MRT bandwidth in low-quality
modes. This is a ~2-day win instead of 2 weeks.

## References

- Burns & Hunt — "The Visibility Buffer: A Cache-Friendly Approach to
  Deferred Shading" (JCGT 2013)
- UE5 Nanite "Deep dive" (GDC 2022) — visibility buffer + cluster cull
- Activision's "Geometry Rendering Pipeline Architecture at Call of Duty"
  SIGGRAPH talks for visibility-buffer variants

## Acceptance

- Sponza fragment bandwidth (measurable via Xcode Metal capture) drops by
  ≥ 50%.
- Same PBR output (SSIM ≥ 0.99 vs baseline).
- Post-FX that consume G-buffer content still work.
- Doesn't regress perf on non-overdraw-heavy scenes.

## Notes for the implementer

- wgpu needs a storage buffer of per-mesh vertex data (triangle index buffer
  + vertex attribute buffer, GPU-indexed by mesh_id). That aligns with
  ticket 009 (GPU-driven rendering).
- Animated meshes (skinned) need special handling — triangle positions change
  per frame. Either compute-skin to a fixed buffer first, or keep animated
  meshes on the traditional path and use visibility buffer for static only.

## Files likely to change

- `native/shared/src/renderer/` (`mod.rs` + `scene_pass.rs`; the old single
  `renderer.rs` was split) — scene_pipeline, main_hdr_pass, shading
  pass, SSR/SSGI/SSAO inputs.
- `native/shared/src/scene.rs` — mesh_id assignment, vertex buffer layout.

## Deferred — reopen criteria

Real GPU bandwidth win (~14 MB/frame at 1600×900 × overdraw factor, on
a benchmark that currently writes 26 MB/pass) but **invisible behind
the vsync cap on Sponza**. The main perf target landed at 60 fps vsync
on full quality, so any further pass-cost reduction just gives headroom
we can't measure here.

Reopen when one of these triggers:

- **A target scene pushes past the 16.7 ms vsync ceiling on the
  benchmark machine.** The 50%+ fragment-bandwidth reduction from a
  visibility buffer is the remaining GPU-side lever for Sponza-class
  bandwidth-bound scenes.
- **Integrated / mobile GPUs become a priority.** Bandwidth matters
  disproportionately more on tile-based and integrated hardware; this
  ticket is the single biggest available reduction.
- **Overdraw-heavy scenes** (foliage, hair, transparent-dense
  particles) become the target. The "every visible pixel shades
  exactly once" property of a visibility buffer + depth-prepass combo
  is essentially the only way to keep overdraw from eating bandwidth.

Effort when reopening is a 2+ week redesign: main_hdr_pass output
becomes `Rgba32Uint (tri_id, u, v, mesh_id)` only, a new shading pass
fetches vertex data from per-mesh storage buffers and evaluates PBR,
downstream MRT consumers (SSR / SSGI / SSAO / post-FX) need to read
from the rebuilt material channels rather than the current 4-MRT
layout. Ticket 005's depth-prepass is a natural prerequisite (it was
deprioritized but would become useful again here). Ticket 009's
unified vertex/index buffers are a hard prerequisite (the shading
pass needs a single bindless-style fetch across all meshes).

The "simpler intermediate step" in the approach section above — drop
unused MRTs when features are off — is a legitimate ~2-day quick win
for low-quality modes (`--quality 1` / `--quality 0` users on
integrated hardware). That's the most-likely first concrete follow-up
when this ticket reopens.
