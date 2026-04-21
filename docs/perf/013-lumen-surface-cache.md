# 013 — Lumen Surface Cache (Mesh Cards)

**Effort:** 1-2 weeks · **Expected gain:** HW bounce reaches full Lumen quality · **Status:** open

Phase 2 of the [Lumen roadmap](lumen-roadmap.md). Depends on 007a + 007b.

## Problem

Ticket 007b's hit-lighting-lite shades BVH intersections with flat per-instance
albedo × sun + skylight. Rich indirect lighting needs per-texel albedo, normal,
and (eventually) roughness at the hit point — but bindless vertex + texture
fetch in WGSL is not available in wgpu in a practical form. UE5's solution is
the **Surface Cache**: each mesh pre-bakes a "card atlas" of albedo / normal /
depth / emissive captured from 6 orthogonal axes, and hits sample this atlas.
The card atlas is re-lit each frame by a compute pass that projects the
shadow cascades + analytic sky onto it. Ray hits then fetch from the already-lit
radiance card.

## Approach

### Card capture — at model load

Per mesh, at `bloom_model_load_glb`:

1. Compute AABB, grow by 5% for edge bleed.
2. For each of the 6 axes (±X, ±Y, ±Z), render an orthographic projection of
   the mesh into a card slot in the **mesh card atlas** — target 256×256 per
   axis face, packed into a 2D texture atlas shared across all loaded meshes.
3. Attachments per card: albedo (Rgba8UnormSrgb), normal-ws (Rg16Snorm
   octahedral-encoded), depth (R16Float, card-local distance), emissive
   (R11G11B10Float).
4. Store the atlas UV rect per mesh-face in a new `MeshCardSlots` SSBO,
   indexed by `mesh_id * 6 + face`.

Use the existing shadow-pass pipeline with a new orthographic projection to
capture albedo; a new lightweight "capture" shader emits the other three
attachments.

### Card lighting — per-frame compute pass

New `card_light_pass` that runs once after the shadow pass and before the
probe trace:

- Input: albedo atlas, normal atlas, depth atlas, shadow-cascade atlas, sun dir,
  sky radiance SH.
- Output: `card_radiance_atlas` (Rgba16Float) — the lit radiance per card texel.
- Per-texel: reconstruct world position from card-face transform + depth,
  NdotL × shadow-sample + sky-SH evaluation × albedo + emissive. Same
  analytical lighting as 007b's hit-lighting-lite, but per-card-texel rather
  than per-instance.

Cost scales with *atlas size*, not scene size — this is the Lumen win. For
100 meshes × 6 faces × 256² ≈ 40 M texels at half-res update (48 bytes/texel
input + 8 bytes output) the bandwidth is modest; the shader is memory-bound.

### Hit shading — swap out hit-lighting-lite

In `SSGI_PROBE_TRACE_HW_WGSL`, replace the inline analytical shading with:

```wgsl
let hit_world = ray_origin + ray_dir * hit.t;
let face = dominant_axis(ray_dir);   // which of 6 cards is most aligned
let slot = mesh_card_slots[hit.instance_custom_index * 6 + face];
let card_uv = project_world_to_card(hit_world, slot);
let radiance = textureSampleLevel(card_radiance_atlas, samp, card_uv, 0.0);
```

Sampling a 2D texture at a computed UV is cheap; the analytical shading work
moves from per-hit (millions) to per-card-texel (millions but amortised —
same texels sampled many times per frame).

SW path (007a) can also sample the card atlas when its screen-space march hits
a visible surface — tighter and more consistent than re-sampling HDR.

### Dynamic-mesh fallback

Meshes flagged dynamic (moving skeleton, morph targets) skip card capture
and fall back to hit-lighting-lite. Card re-capture per frame for dynamic
meshes is out of scope; revisit when dynamic Sponza meshes appear.

## Files likely to change

- `native/shared/src/models.rs` — card capture hooks at load.
- `native/shared/src/renderer/mod.rs` — card atlas textures, capture pipeline,
  `card_light_pass`, probe-trace HW binding additions.
- `native/shared/src/renderer/shaders.rs` — `CARD_CAPTURE_WGSL` (new),
  `CARD_LIGHT_WGSL` (new), `SSGI_PROBE_TRACE_HW_WGSL` update.
- `native/shared/src/scene.rs` — card-slot metadata on scene nodes.

## Acceptance

- Visual quality: HW bounce on Sponza column interiors matches the fidelity
  of the reference UE5 Sponza Lumen capture (qualitative — see commit PNGs).
- Textured bleed: a yellow wall behind the camera should cast a yellow tint
  into the visible scene. Before: flat white tint (hit-lighting-lite).
  After: warm yellow.
- Performance: `--quality 3 --ssgi 1 --fps-only 300` ≥ 45 fps (card-light
  pass adds ~2 ms; probe trace savings offset it). Worst case: flat.
- Card atlas size stays under 64 MB VRAM for Sponza (68 materials).

## Notes

- Atlas packing: start with a naive grid (every mesh gets 6 equal slots);
  optimise only if VRAM blows out. Lumen uses adaptive resolution based on
  screen footprint; that's out of scope for v1.
- Card capture runs at model load, not per frame — the cost is paid once.
- The card atlas is a natural home for future bounces (sample the previous
  frame's card radiance when lighting the current frame → infinite bounces).
  Flag as follow-up.
