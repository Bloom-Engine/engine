# Temporal reactive coverage

Bloom's TAA history is now coverage-aware for imported glTF `alphaMode: BLEND`
and contributing `KHR_materials_transmission`. Without that signal, a moving
or changing transparent layer can reproject valid opaque velocity while its
color still depends on a different current-frame layer or background sample.
The result is stale color trails around glass, foliage-like BLEND cards, and
refracted silhouettes.

## Coverage contract

When TAA and at least one visible imported contributor are active, the compiled
frame graph creates one render-resolution transient:

- `transparency-reactive`: `r8unorm`, color-attachment plus sampled usage.

Imported passes union their coverage into it:

```text
combined = source + destination * (1 - source)
```

This is exact front-to-back coverage union and remains bounded in `[0, 1]`.
Sorted BLEND writes its shaded base alpha. Weighted OIT writes resolved opacity
`1 - revealage`. Opaque physical transmission writes its transmission weight;
BLEND+transmission writes base alpha because the complete resolved material
response is mixed by that factor.

The TAA variant samples coverage at the same unjittered UV as current color and
uses it as a lower bound on current-frame weight:

```text
history_alpha = max(motion_alpha, disocclusion_alpha, reactive_coverage)
```

A 20% transparent contribution therefore rejects at least 20% stale history,
while fully refractive pixels consume the current result immediately. Linear
mask filtering preserves sub-pixel edge coverage at native and upscaled output
resolutions.

## Lazy-path guarantees

The mask and every pipeline that writes or reads it are lazy. The established
pipelines remain selected without modification when any of these is true:

- the frame has no visible imported BLEND or transmission draw;
- TAA is disabled;
- translucency consists only of user-authored custom material buckets;
- weighted OIT is active while TAA is disabled.

Consequently opaque frames retain the existing TAA bind-group ABI and shader,
perform no extra texture sample, allocate no mask, and compile no reactive
shader. TAA-active imported frames pay one byte per render pixel. A mixed
imported/custom sorted frame remains one globally ordered render pass. Bloom
lazily creates an attachment-compatible custom pipeline sibling whose second
target has an empty write mask, so custom material source/ABI and HDR output
stay unchanged while only imported draws write coverage. The sibling is not
compiled for opaque, custom-only, TAA-off, weighted, or unmixed sorted frames.

Set `BLOOM_TEMPORAL_REACTIVE=0` before renderer creation for an exact A/B
diagnostic. This restores the previous graph topology and all established
transparency/TAA pipelines; it is not intended as a shipping quality setting.
Qualification telemetry reports `enabled`, `active`, format, and graph
allocation counts under `renderer_paths.temporal_reactive`.

## Qualification

The `quality-transparency` fixture supports all three GPU routes without
changing its default weighted corpus:

```sh
# Existing 96-layer weighted OIT corpus.
./main --quality-run ...

# Force the sorted imported-BLEND writer.
./main --sorted --quality-run ...

# Use the physical transmission fixture and refractive writer.
./main --refractive --quality-run ...
```

Required regression evidence is:

- ordinary opaque goldens remain unchanged;
- TAA-off and `BLOOM_TEMPORAL_REACTIVE=0` plans contain no
  `transparency-reactive` resource;
- active sorted, weighted, and refractive routes pass backend validation;
- a mixed custom/imported sorted route preserves global depth order with TAA
  both off and on, and the custom sibling does not modify the R8 mask;
- changes between reactive on/off are localized to imported transparent
  coverage and improve motion-history behavior;
- the extra allocation is exactly one render-resolution byte per pixel and
  active-frame CPU/GPU budgets remain satisfied.

## Current boundary

Custom `Transparent`, `Refractive`, and `Additive` material buckets do not yet
declare temporal coverage. Their existing ABI is deliberately preserved rather
than silently widening every user material pipeline. A future material API can
add an explicit reactive output without changing the imported glTF contract.
