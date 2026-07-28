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

## Platform device negotiation

Every native startup path selects one of these profiles and requests its actual
active shader-layout contract instead of `wgpu::Limits::default()` or the
adapter's entire advertised budget:

| Profile | Platforms/layout | Bind groups | Color attachments | Sampled textures/stage | Samplers/stage | Storage buffers/stage | Uniform binding |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `native-full` | macOS, iOS/tvOS/visionOS, Linux, Windows; separate SceneInputs group | 5 | 4 | 19 | 16 | 8 | 64 KiB |
| `folded-mobile` | Android; SceneInputs folded into group 0 | 4 | 4 | 19 | 16 | 4 | 64 KiB |

Android's lean main pass uses two color attachments, but its platform contract
remains four because other compiled pipelines may use up to four. Path tracing
raises the storage-buffer requirement to 9. Tier A requests only Bloom's
bounded 4,096-texture/64-sampler working set, including fallback entries,
rather than the adapter's maximum descriptor budget.

Startup first requests supported optional features for the selected tier. If
the backend rejects that request, Bloom retries the same active shader-layout
contract without optional features, selecting the resulting modern/baseline
resource tier. Both success and fallback emit a structured report; the chosen
request and the preferred-request failure cause are also embedded in quality
artifacts under `adapter.device_negotiation`.

An adapter below the active shader-layout contract receives an explicit error
naming every insufficient limit. It is not reported as a successful lower tier
until Bloom has an implementation of that lower shader layout.
