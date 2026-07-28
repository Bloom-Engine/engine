# Renderer capability tiers

Bloom selects renderer paths from granted device features and limits, never from
an operating-system or GPU-name allowlist. `BLOOM_FORCE_RENDER_TIER` accepts
`baseline`, `modern`, or `high-end` and can force a supported lower tier for
qualification. An unsupported upward request is rejected and reported.

The table below is validated byte-for-byte against the definitions used by the
runtime. Independent optional features remain feature-detected inside a tier:
automatic selection does not turn off GPU-driven submission or ray query merely
because an unrelated optional feature is absent.

<!-- BEGIN GENERATED CAPABILITY TIERS -->
| Tier | Materials | Geometry | Shadows | GI | Reflections | AA | Textures | Path tracing | Minimum contract |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Tier C per-material bind groups | CPU direct draws | Cascaded/VSM raster paths | Software SDF, probes, and SSGI | SSR, planar, and probe fallbacks | TAA/CAS/FXAA | Per-material resident textures | Disabled when this tier is forced | Active platform profile |
| modern | Tier B deterministic paged arrays | CPU direct draws | Cascaded/VSM raster paths | Software SDF, probes, and SSGI | SSR, planar, and probe fallbacks | TAA/CAS/FXAA | Paged texture arrays/atlases | Disabled when this tier is forced | 16 texture-array layers; 8 sampled textures/stage |
| high-end | Tier A descriptor-indexed global tables | GPU indirect when supported; CPU oracle fallback | Cascaded/VSM raster paths | Ray query when supported; software SDF/SSGI fallback | Ray query when supported; SSR/planar/probe fallback | TAA/CAS/FXAA | Descriptor-indexed texture/sampler arrays | Available only with ray query and required limits | Texture-binding arrays + non-uniform indexing; 2 array elements |
<!-- END GENERATED CAPABILITY TIERS -->

This table describes the cross-system path contract after device creation.
Platform profiles still own their minimum surface and shader-layout limits.
Those startup contracts and lower-tier device-request retries are tracked as the
next capability-tier migration step.
