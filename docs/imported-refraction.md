# Imported glTF refraction

Bloom routes contributing `KHR_materials_transmission` materials through a
dedicated forward bucket. The route is selected at renderer startup and is
reported by `getImportedRefractionMode()`:

- `scene-snapshot`: desktop/native targets sample immutable pre-translucency
  HDR color and depth supplied by the compiled render graph;
- `environment-fallback`: WebGPU and Android preserve the dielectric material,
  IOR, texture modulation, Fresnel response, thickness, and absorption, but
  source transmitted radiance from the environment because those targets fold
  the renderer to four bind groups;
- `disabled-legacy`: diagnostic opt-out selected by setting
  `BLOOM_GLTF_REFRACTION=0` before renderer and asset creation.

The default is physical refraction. The legacy mode exists only as an A/B kill
switch and restores the previous mirror-like import approximation.

## Shading contract

The physical material bind group is allocated only for an active authored
transmission factor. Ordinary materials keep their existing material layout,
shader, graph topology, and pass list.

The shader applies:

1. transmission/thickness texture modulation, including
   `KHR_texture_transform` and independent `TEXCOORD_0`/`TEXCOORD_1`
   selection for each texture;
2. Snell refraction with the authored/default IOR;
3. a depth-guarded, 64-pixel-bounded screen-space offset on snapshot-capable
   targets;
4. deterministic rough-transmission filtering;
5. Beer-Lambert attenuation using thickness, attenuation color, and
   attenuation distance;
6. Schlick Fresnel partitioning between transmitted and reflected energy,
   using the bounded planar/screen-space/environment hierarchy on native
   targets and the prefiltered environment on folded targets.

Model scale promotes authored thickness into world units. The pass writes
per-object motion vectors, depth-tests against opaque geometry, and does not
write opaque depth. With TAA active, it also writes transmitted contribution
to the lazy `r8unorm` reactive target so background-dependent radiance cannot
leave stale temporal trails. See
[Temporal reactive coverage](temporal-reactive-coverage.md).
The reflection source and its exact fallback/allocation contract are described
in [Refractive reflection sources](refractive-reflections.md).

### Secondary-UV cost contract

`Vertex3D` remains the established 96-byte ordinary vertex ABI and every
opaque, MASK, BLEND, UV0-refraction, depth, GI, and ordinary shadow pipeline
keeps its original vertex layout. When an active transmission or thickness
texture actually requests `TEXCOORD_1`, the importer retains a compact
8-byte-per-vertex sidecar. The renderer uploads and fetches that sidecar only
when the referenced texture is usable, then selects a separately compiled
two-stream refraction pipeline.

Transmission and thickness select UV0 or UV1 independently before applying
their own `KHR_texture_transform`; mixed-UV materials therefore do not need
duplicated vertices. Cached meshes localize their shared-arena primary/index
windows so the mesh-local sidecar uses the same zero-based indices. Retained,
cached, cached-skinned, directional-shadow, and TAA-reactive routes share this
contract.

An unreferenced `TEXCOORD_1` accessor incurs no retained CPU data, GPU buffer,
second vertex fetch, or extra pipeline. A requested but missing/malformed UV1
accessor is never synthesized from UV0: Bloom preserves the source binding,
diagnoses the mismatch, and uses the scalar physical factor.

## Composition and shadows

Retained and cached imported materials share one deterministic back-to-front
key: view depth followed by stable object ID. The shader composites against an
immutable snapshot and writes a fully resolved HDR result, preventing
read/write feedback and double application of the background.

Physical-transmission materials cast bounded colored directional shadows
without entering the opaque CSM. The existing 2048² opaque/MASK cascades remain
authoritative for blockers. When the scene actually submits an eligible
transmission caster, Bloom lazily adds one 1024² RGB transmittance texture and
one 1024² nearest-layer depth texture per cascade. The resolve subtracts only
the absorbed portion of the already-shaded primary sun contribution, including
opaque and cloud visibility.

Ordinary scenes do not allocate these textures, compile the caster/resolve
pipelines, add graph resources, or add a pass. Set
`BLOOM_TRANSMITTED_SHADOWS=0` before renderer creation for an exact A/B opt-out.
The selected policy and live caster count are exposed in native quality
telemetry. See [Transmitted directional shadows](transmitted-shadows.md).

## Global illumination

With screen-probe GI enabled, retained physical-transmission instances use one
bounded colored continuation rather than behaving as opaque GI blockers.
Hardware ray query performs at most one extra opaque-only query when glass is
the nearest hit. The software SDF route keeps glass out of the opaque clipmap
and selects one nearest conservative transmission AABB from existing instance
metadata.

The route adds no GPU resource, graph pass, or transient slot and creates its
specialized pipelines only after an eligible retained instance appears. Set
`BLOOM_TRANSPARENT_GI=0` for the exact opaque-GI A/B control. See
[Transparent global illumination](transparent-gi.md).

## Explicit limits

- Physical textures authored against `TEXCOORD_2` or higher remain preserved
  in metadata, are diagnosed at material creation, and fall back to their
  scalar factors instead of sampling the wrong coordinates.
- The environment fallback cannot show on-screen background distortion.
- V1 native reflection routing consumes only the first explicit planar probe;
  non-matching glass safely falls through to screen space/environment.
- The native reflection hierarchy deliberately does not launch a per-fragment
  ray query on screen misses; see the bounded-cost rationale in
  [Refractive reflection sources](refractive-reflections.md).
- Arbitrarily deep nested/order-independent dielectric refraction is outside
  this baseline.
- The directional-shadow representation keeps the nearest transmission layer
  per light texel. Arbitrarily nested transparent shadow casters are outside
  this bounded baseline.
- Transparent GI keeps one nearest transmission instance per ray. Its software
  fallback uses a conservative instance AABB, and GI transport does not sample
  per-hit transmission/thickness textures.
