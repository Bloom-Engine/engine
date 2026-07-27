# Transparency composition

Bloom keeps two conventional imported glTF `alphaMode: BLEND` composition
routes. Neither route writes opaque depth or participates in the opaque shadow
and planar-probe lists.

## Modes

Use `setTransparencyCompositionMode()`:

- `"sorted"` always uses deterministic back-to-front alpha blending.
- `"auto"` is the default. It keeps sorted blending below 64 visible imported
  BLEND draws and selects weighted-blended OIT at or above that threshold.
- `"weighted"` forces weighted-blended OIT whenever at least one imported BLEND
  draw is visible. Use this for intersecting surfaces whose relative depth
  order changes across one primitive or during camera motion.

`getTransparencyCompositionMode()` reports the configured policy.
`getActiveTransparencyCompositionMode()` reports `"sorted"` or `"weighted"`
for the most recently prepared frame. `BLOOM_TRANSPARENCY=sorted|auto|weighted`
sets the startup policy; the API can change it later.

The auto threshold is intentionally conservative. Sorted alpha is exact for
simple, non-intersecting layers, while weighted OIT trades exact layer order
for bounded, stable composition when object sorting cannot represent the
per-pixel order.

## Weighted path

The accumulation pass uses two render-graph-owned, render-resolution transient
targets:

- `transparency-accumulation`: `rgba16float`;
- `transparency-revealage`: `r16float`.

For source radiance `C`, opacity `a`, and normalized fragment depth `z`:

```text
w         = 0.1 + 0.9 * (1 - z)^3
accum.rgb += C * a * w
accum.a   += a * w
reveal    *= 1 - a
```

The resolve computes:

```text
opacity = 1 - reveal
color   = accum.rgb / max(accum.a, 1e-5)
HDR     = color * opacity + HDR * (1 - opacity)
```

The bounded weight avoids half-float overflow from the large exponential
weights used by some WBOIT variants. A single layer is algebraically identical
to conventional alpha blending. Accumulation and revealage blending are
order-independent; retained-scene and cached-model imported draws share one
visible draw list and stable ID contract.

The targets, bind group, shader module, and three pipelines are lazy. Opaque
frames and sorted-only transparency plans do not allocate or compile them.
Bind groups are cached by compiled graph plan plus transient resize generation.

## Global sorted path

Conventional imported BLEND draws and user-authored custom `Transparent`,
`Refractive`, and `Additive` commands share one back-to-front dispatcher when
weighted OIT is inactive. The key is:

1. view depth, far to near;
2. source rank only at exactly equal depth (imported before custom); and
3. the source's stable object/submission ID.

Retained and cached imported draws keep their existing common stable-ID space.
Custom commands keep submission order at equal depth. Switching source changes
pipeline/bind-group state but not the composition order.

With TAA reactive coverage active, custom material source/ABI remains
unchanged. On the first frame that actually mixes custom and imported sorted
draws, Bloom lazily compiles an attachment-compatible sibling of each
participating custom pipeline. It preserves the exact HDR blend/depth/shader
contract and declares the R8 reactive target with an empty write mask. Imported
draws can therefore union real coverage while custom draws in the same render
pass leave it untouched. This replaces the former imported-then-custom pass
split without adding a draw, graph pass, or image.

Set `BLOOM_SORTED_INTERLEAVING=0|false|off|disabled` before renderer creation
to restore the previous imported-list-then-custom-list behavior for exact A/B
diagnosis. With TAA active, that control also restores the former two-render-
pass attachment split. It is a rollback/qualification switch, not a shipping
quality mode.

## Current boundaries

- Physical `KHR_materials_transmission` uses the separate imported-refraction
  pass and is not folded into weighted OIT. Arbitrarily nested,
  order-independent refraction remains a non-goal.
- User-authored custom material `Transparent`, `Refractive`, and `Additive`
  buckets are globally interleaved with imported BLEND only in sorted mode.
  When imported WBOIT is active, its aggregate resolve occurs before those
  custom commands. Interleaving a per-draw custom command inside an
  order-independent imported aggregate has no well-defined object order.
- With TAA active, resolved opacity is unioned into the lazy
  `r8unorm` temporal-reactive target. See
  [Temporal reactive coverage](temporal-reactive-coverage.md). TAA-disabled
  weighted frames keep the established resolve and allocate no mask.
- The route uses two full-resolution transient targets only while active:
  10 bytes per render pixel before allocator alignment. TAA-active imported
  transparency adds one byte per render pixel for reactive coverage.

## Qualification contract

Changes to this path must retain:

- an order-sensitive sorted negative control;
- an AB/BA intersecting imported-BLEND GPU test with weighted mean RGB
  difference at most 0.02/255 and maximum channel difference at most 2/255;
- a simple-BLEND test proving auto mode stays sorted and the graph contains no
  weighted targets;
- a mixed imported/custom property test proving moving the custom layer from
  behind to in front changes the overlapped result under the global key;
- a TAA-active mixed test proving the lazy custom sibling validates and leaves
  imported reactive coverage untouched;
- unchanged opaque goldens and an opaque quality capture with zero weighted
  transient allocations;
- a TAA-active run proving the reactive target adds exactly one byte per render
  pixel, plus a `BLOOM_TEMPORAL_REACTIVE=0` negative control;
- native, Web/model, and Android/model compile checks.
