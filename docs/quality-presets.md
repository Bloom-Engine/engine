# Quality presets

Bloom keeps its raylib-style individual switches, but a preset is a complete
starting policy rather than a bag of effect toggles. Resolution, temporal
reconstruction, upscale filtering, sharpening, and optional effects move
together:

| Preset | Render scale | TAA | Upscale | Composite sharpen | Main effects |
|---|---:|---|---|---:|---|
| Off | 0.50 | Off | Bilinear | 0.00 | Base HDR/tonemap only |
| Low | 0.67 | Off | Catmull-Rom | 0.25 | Bloom |
| Medium | 0.75 | On | Catmull-Rom | 0.40 | Shadows, SSAO, bloom |
| High | 0.85 | On | Catmull-Rom | 0.45 | Medium + SSR, SSGI, subtle chromatic aberration |
| Ultra | 1.00 | On | Native | 0.85 | Full effect stack |

`setQualityPreset()` applies the row as one operation. Call individual setters
afterward to override it:

```typescript
import {
  QualityPreset,
  setQualityPreset,
  setMotionBlurEnabled,
  setRenderScale,
} from "@bloomengine/engine/core";

setQualityPreset(QualityPreset.Ultra);
setMotionBlurEnabled(false);

// Or retain High's effect policy while choosing a custom resolution.
setQualityPreset(QualityPreset.High);
setRenderScale(0.90);
```

## First-run default

A renderer that has not received a preset starts at render scale `0.75`, TAA
on, Catmull-Rom upscale, composite sharpen `0.50`, and no extra CAS pass. The
former `0.50` default shaded only one quarter of the output pixels and read as
broken image quality at ordinary window sizes. The new default shades 56.25%
of output pixels, while Off and Low preserve explicit performance tiers.

Render resolution and anti-aliasing are independent. `setTaaEnabled()` never
changes `render_scale` and does not rebuild resolution-dependent targets.
Use `setRenderScale()` for resolution, or enable dynamic resolution when a
fixed frame-rate target matters more than a fixed scale.

## Sharpening

The preset table uses Bloom's existing composite-pass unsharp mask, so Medium
through Ultra do not add a render pass. The separate contrast-adaptive sharpen
pass remains opt-in through `setCasStrength()` and defaults to zero in every
preset. This avoids paying for two sharpen stages or producing double halos.

At sub-native scale, texture mip bias, TAA sample weighting, and projection
jitter already follow the selected render extent. Catmull-Rom is used for Low,
and the TAA resolve performs temporal upscaling for Medium and High. Ultra
renders at native resolution and uses TAA only for anti-aliasing and
sub-pixel accumulation.
