# Transmitted directional shadows

Bloom represents directional shadows from imported physical-transmission
materials as a lazy, bounded correction layered over the existing cascaded
shadow map (CSM). The opaque and MASK route remains unchanged and authoritative;
glass never becomes an opaque blocker merely to produce a shadow.

## Activation and cost contract

The route activates only when all of the following are true:

1. imported physical transmission is enabled;
2. directional shadows are enabled;
3. at least one visible retained or cached transmission draw is eligible to
   cast a shadow; and
4. `BLOOM_TRANSMITTED_SHADOWS` is not set to `0`, `false`, `off`, or
   `disabled`.

Before activation there is no allocation, pipeline creation, transmitted-shadow
graph resource, or resolve pass. Activation allocates, once:

| Resource | Cascades | Extent | Format | Bytes |
| --- | ---: | ---: | --- | ---: |
| transmittance | 3 | 1024² | RGBA8Unorm | 12,582,912 |
| nearest depth | 3 | 1024² | Depth16Unorm | 6,291,456 |
| **total** | | | | **18,874,368 (18 MiB)** |

These are persistent imported graph resources, not frame-transient textures.
They therefore add zero transient slots and cannot alias a live frame target.
The exact size is asserted in a unit test and reported by native quality
telemetry.

## Caster representation

Each cascade stores the transmittance and depth of the nearest eligible
transmission surface. Nearest-layer depth deliberately rejects the far face of
a closed volume, preventing the same authored thickness from being absorbed
twice. It also establishes a deterministic upper bound: overlapping or nested
glass does not grow storage or fragment work with layer count.

The caster evaluates the same material inputs as camera-facing imported
refraction:

- base color, vertex color, instance tint, and BLEND coverage;
- MASK cutoff;
- metallic suppression of dielectric transmission;
- transmission factor and texture;
- thickness factor and texture;
- attenuation color and distance;
- IOR Fresnel partitioning; and
- model scale for world-space thickness.

Transmission and thickness textures use the same independent UV0/UV1
selection and `KHR_texture_transform` contract as camera-facing refraction.
The 8-byte TEXCOORD_1 vertex stream and its caster pipeline are created only
after a usable physical texture requests UV1; UV0 casters retain the original
single-stream pipeline and vertex fetch.

The established opaque cascade is sampled before writing. A transmission
surface behind a nearer opaque blocker cannot tint that blocker's shadow.
Retained, cached, and cached-skinned draws share this route; the latter reuses
the existing joint palette.

## Receiver resolve

The optional `transmitted_shadow_resolve` graph pass runs after path tracing
and before translucent composition. It reconstructs receiver world position
from scene depth, manually bilinearly resolves the transmittance/depth maps,
blends across the established cascade transitions, and reconstructs the
primary directional-light PBR term from the material G-buffer.

The pass additively writes only the negative correction:

```text
direct_sun * opaque_visibility * cloud_visibility * (transmittance - 1)
```

This avoids multiplying emissive, ambient, image-based, GI, or later
translucent lighting. It also preserves the existing opaque CSM and shared
world-space cloud visibility. Because Bloom has no normal G-buffer, receiver
normal is reconstructed from neighboring world positions; normal-map detail
is not part of this bounded resolve.

## Caching and invalidation

A cascade is rerendered only when its light view-projection, deterministic
caster/material/transform signature, or corresponding live opaque-depth
generation changes. Static glass and blockers therefore reuse the persistent
maps. Moving cached-skinned draws invalidate conservatively.

The caster budget is capped at 1024 submitted draws and diagnoses overflow
once. Casters outside each cascade are rejected by a conservative world-space
AABB/frustum test.

## Explicit limits

- Only the nearest transmission layer contributes per light texel. Deep,
  order-independent transparent shadow stacks are intentionally outside this
  baseline.
- The map resolution is half the opaque CSM linear resolution.
- The resolve reconstructs geometric receiver normals rather than normal-map
  detail.
- This route covers the primary directional light; local-light transmission
  shadows remain a separate representation problem.

## Qualification

The live Metal stress A/B used 96 moving physical-transmission layers at
960×540, with 60 warm-up and 120 measured frames. The feature-on graph compiled
once, reused one cached plan for 178 frames, and retained the existing three
transient slots. The two added profiler regions measured approximately
0.95 ms (`transmitted_shadow_maps`) and 0.69 ms
(`transmitted_shadow_resolve`) mean GPU time in that deliberately saturated
scene. Feature-on/off GPU p95 was within measurement noise.

The captured on/off images differ only in the localized transmitted tint:
SSIM 0.9999938, luminance RMSE 0.0002146, maximum channel error 0.01961, and
zero pixels above the 0.02 tolerance. A live GPU property test additionally
toggles only `cast_shadow` and verifies both a non-trivial affected region and
the authored RGB attenuation ordering.

Native telemetry under `renderer_paths.transmitted_shadows` exposes:

- `enabled` and `active`;
- `representation` (`nearest-layer-rgb-depth`);
- `resolution`;
- `persistent_bytes_when_allocated`; and
- `caster_count`.
