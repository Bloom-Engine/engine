# Refractive reflection sources

Native imported physical transmission uses a bounded reflection-source
hierarchy:

1. an explicit planar probe for glass lying on that probe's plane;
2. a glass-local screen-space ray against the immutable opaque color/depth
   snapshots;
3. the prefiltered environment map.

This path is separate from opaque SSR. The opaque SSR texture was traced from
the opaque surface behind the glass and therefore has the wrong origin and
normal for a dielectric fragment. Reusing it would produce plausible-looking
but geometrically incorrect reflections.

## Source rules

The planar tier reuses the first active `createPlanarReflection` capture. A
fragment accepts it only when:

- its world position is within 0.075 world units of the probe plane;
- its unperturbed normal is aligned to that plane; and
- non-zero probe alpha contributes captured geometry; alpha zero is the
  established geometry-miss marker and reveals the prefiltered environment.

Planar captures have one mip level. Their contribution fades from full weight
at roughness 0.18 to the lower tiers at roughness 0.45, avoiding an
incorrectly sharp rough reflection.

Glass without an applicable planar probe launches eight quadratically spaced
samples up to eight world units through the existing pre-translucency depth
snapshot. A depth-thickness confidence test rejects crossings, an eight-pixel
boundary fade suppresses screen-edge popping, and the result fades to the
environment by roughness 0.45. Misses, invalid samples, disabled SSR, and rough
surfaces all return the prefiltered environment.

The hierarchy does not issue a per-fragment ray query. The retained TLAS does
not contain every immediate draw, and shading an arbitrary query hit would
require another material/radiance representation. Adding that work only on
screen misses would create scene-dependent performance cliffs. Hardware ray
query remains available to the bounded transparent-GI path, where its
representation and cost are explicit.

## Allocation and fallback contract

The native hierarchy reuses the color/depth snapshots already required by
physical refraction and any explicitly created planar probe. It adds:

- no graph resource, pass, image, transient slot, or image byte;
- one lazy 160-byte uniform after the first physical-transmission material;
- one dedicated native group-4 layout and bind group for that material path.

Ordinary scene materials and the folded Web/Android four-group shaders are
unchanged. Folded targets retain their prefiltered-environment reflection.
With no physical-transmission draw, the hierarchy creates no layout, uniform,
bind group, or shader variant.

Set `BLOOM_REFRACTIVE_REFLECTIONS=0` (also `off`, `false`, or `disabled`)
before renderer creation to compile the exact prior environment-only
reflection expression and skip every hierarchy resource. Runtime
`setSsrEnabled(false)` disables the screen-space tier while leaving a matching
explicit planar probe available.

Native quality telemetry reports `renderer_paths.refractive_reflections`,
including enable/active state, selected source order, fixed march bounds,
lazy persistent bytes, and zero graph/image cost.

## Planar depth convention

Planar captures use an oblique near plane. wgpu/Metal/D3D depth is
`0 <= z <= w`, so the projection's replacement z row is
`plane / dot(plane, far_corner)`. The OpenGL `[-w,w]` formula maps the plane to
`z=-w`; using it on wgpu incorrectly culls and clips valid above-plane
geometry. The projection helper and its regression test pin the native
`z=0` plane invariant.

## Qualification route

`examples/quality-transparency/main.ts --reflection-hierarchy` is an
unversioned focused oracle. It places smooth imported transmission on an
explicit horizontal probe and reflects a rotating Damaged Helmet. The default
versioned weighted-transparency corpus is unchanged.

V1 deliberately selects only the first active planar probe. Glass on another
plane safely falls through to screen space/environment; automatic per-material
multi-probe selection remains future work.
