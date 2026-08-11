# Masked alpha coverage

Imported glTF materials with `alphaMode: MASK` preserve their authored
silhouette as the base-color texture minifies. This is a material-specific
import path: OPAQUE, BLEND, normal-map, and standalone texture mip chains keep
their established bytes and runtime behavior.

## Import contract

For each MASK material that has a base-color texture, Bloom computes the
cutoff in texture-alpha space:

```text
coverage reference = alphaCutoff / baseColorFactor.a
```

The importer registers a cutoff-specific texture variant. This matters when
multiple MASK materials share one source image but use different cutoffs.
Level zero remains byte-for-byte identical to the decoded source. Lower levels
store:

- alpha: the fraction of source texels that survive the effective cutoff;
- RGB: the linear-light, coverage-weighted mean of surviving texels.

Reduction integrates the exact source footprint, including odd rows and
columns in non-power-of-two images. Visible colors dilate into empty texels
after reduction, so bilinear filtering cannot pull transparent border colors
into foliage and fence silhouettes. References above one intentionally yield
zero lower-mip coverage, matching a base-color factor that can never reach the
material cutoff.

Bloom also follows glTF's multiplication rule for this path: `COLOR_0`
multiplies `baseColorFactor`; it does not replace it. That keeps imported
level-zero alpha and the factor used to derive the coverage reference in
agreement.

Only imported MASK base-color textures allocate variants, and only when the
target supports lower mips. An ordinary copy of a shared image remains
available when OPAQUE, BLEND, or another texture semantic uses it. A MASK-only
image aliases its first coverage variant instead of retaining an unreachable
ordinary chain. The storage cost is therefore one chain per unique
`(image, effective cutoff)` pair, minus the ordinary chain it replaces for the
common single-cutoff MASK-only case; it adds no render pass or ordinary frame
allocation.

## Raster contract

At magnification and near level zero, scene color, depth prepass, velocity,
and shadow passes retain the exact authored alpha test. At minification they
interpret lower-mip alpha as coverage probability and compare it against the
same deterministic 4×4 Bayer threshold. The threshold is anchored to authored
base-texture texels rather than screen or shadow-map pixels, so its binary
silhouette follows object motion, survives camera reprojection, and does not
re-roll when a shadow cascade refits. A short transition between level zero
and the first coverage mip avoids a discrete LOD pop.

The main and depth-prepass paths use the same texture LOD bias. Shadow cutouts
use the same coverage rule and include vertex alpha, so the caster silhouette
matches the visible geometry. Screen-space stability is handled by the
existing TAA and velocity paths; masked rendering does not introduce a new
history or full-screen pass.

Bloom's current scene, depth, velocity, and shadow targets are single-sample.
Hardware alpha-to-coverage therefore is not available and is not emulated or
reported as supported. The coverage-mip/Bayer path is the explicit
single-sample fallback. Quality telemetry records:

- the number of registered coverage-mip variants;
- whether coverage mips are supported on the target;
- render sample count and alpha-to-coverage support;
- the active single-sample fallback name.

## Platform and authoring boundaries

Android intentionally retains its established one-mip upload path because
multi-mip color uploads are not yet qualified there. It receives the exact
hard-cutout path and does not claim coverage-mip support.

For same-binary qualification, `BLOOM_MASK_COVERAGE=0` disables coverage
variant creation at import time and routes MASK materials through the
established ordinary mip chain. This is a diagnostic A/B control, not the
default; `off`, `false`, and `disabled` are accepted aliases. The check runs
only while importing glTF data and adds no normal-frame work.

Coverage variants bake the imported `baseColorFactor.a` and cutoff. Per-vertex
alpha and runtime draw tint remain exact at level zero but cannot be baked into
the shared lower mip chain. Distant coverage is therefore based on the
imported material and texture, which is the stable authoring contract. Assets
that require independently animated opacity should use BLEND rather than
MASK.

## Qualification

The `masked-alpha-coverage` quality case renders 48 imported MASK cards over a
shadow receiver across six projected mip ranges. It keeps fixed-step object
motion, TAA, and cascaded shadows active, captures HDR/depth/cascade
intermediates, and requires masked-path telemetry. A real-GPU negative control
compares the coverage variant with the ordinary box-filtered mip chain and
asserts that the variant retains the authored 75% silhouette area instead of
filling the card.
