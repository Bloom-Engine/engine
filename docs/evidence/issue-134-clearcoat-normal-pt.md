# Issue #134 path-traced clearcoat-normal evidence

This checkpoint closes the remaining independent-clearcoat-normal gap in
Bloom's layered GPU path tracer. It is qualified at revision `5c0ab20` on an
Apple M1 Max / Metal adapter with ray query and texture arrays enabled.

## Transport contract

The path tracer now preserves the imported clearcoat-normal texture, UV set,
`KHR_texture_transform`, and normal scale at primary and bounce intersections.
It reconstructs the committed vertex tangent and handedness only for a
qualified anisotropy or clearcoat-normal record, maps the coat normal without
changing the base normal, and uses the mapped normal for:

- direct-light next-event evaluation;
- clearcoat importance sampling;
- clearcoat PDFs and MIS;
- reciprocal attenuation of the undercoat;
- subsequent-bounce surface state.

The mapped normal is constrained to the base geometric hemisphere. The
normal-length/variance stored by the existing vector-mip upload path widens
the coat roughness for minification. Path tracing deliberately does not use
screen derivatives; realtime shading additionally retains its established
screen-space curvature variance.

On adapters without the complete texture-array feature pair, only the normal
map is ignored. Scalar clearcoat factor and roughness remain active, and no
normal or UV1 sidecar is allocated.

## Lazy resource cost

Clearcoat-normal metadata has its own 48-byte-per-instance sidecar:

- 16-byte header;
- 2x2 UV transform matrix;
- UV offset and normal scale.

It uses group 2 binding 8, resource-key bit 7, and pipeline-key bit 9. The
existing 64-byte clearcoat factor/roughness record and 96-byte scalar layered
record do not grow. Base, scalar-clearcoat, and factor/roughness-textured
scenes report:

- `path_tracing_clearcoat_normal_specialization_initialized = false`;
- `path_tracing_clearcoat_normal_sidecar_allocated_bytes = 0`.

A normal-only clearcoat material allocates no factor/roughness texture
sidecar. There is no added render-graph pass, image, base-material branch, or
base-pipeline binding. The shared sidecar upload helper retains the established
power-of-two capacity, dirty-upload, and bind-group invalidation behavior.

## Metal image gates

The release ray-query golden runs the complete texture-array path rather than
the former scalar-fallback path. Its fixture owns real per-face UVs locally,
so unrelated transparent-GI geometry remains unchanged.

The zero-scale flat normal is byte-identical to scalar clearcoat. A directional
normal map produces a bounded response without mean display-energy gain:

| Comparison | Mean RGB difference | Maximum channel difference | SSIM |
|---|---:|---:|---:|
| Flat vs directional normal | 0.635595 | 146 | 0.980410 |
| Directional vs 90-degree UV rotation | 0.538315 | 150 | 0.985408 |
| Directional UV0 vs UV1 | 0.601186 | 154 | 0.983375 |

Flat and mapped mean display luminance are `84.161719` and `83.784545`
respectively. The existing layered transport gate supplies the finite,
visible-response, and bounded-energy checks; its thresholds were not weakened.
The rotation and UV1 controls retain their established minimum mean-RGB
difference of `0.02`.

Enabling texture arrays exposed a previous golden-harness blind spot: the
device requested ray query but not its supported texture-array feature pair,
so older textured-lobe assertions always exercised the documented fallback.
The harness now requests both supported features. All textured specular,
clearcoat, sheen, iridescence, and anisotropy variants consequently execute on
Metal. The anisotropy fixture was strengthened to full authored strength and a
clear directional texture while retaining its original visibility threshold.

## Regression qualification

`./scripts/ci-check.sh --quick` passed in 23 seconds:

- 328 shared unit tests passed, 1 intentionally ignored;
- 57 runnable GPU goldens passed, 2 intentionally ignored;
- all 4 render-target integration tests passed;
- format, strict lint policy, FFI/schema parity, web arity, wasm32
  compilation, quality governance, visual fault controls, and canonical
  example inventory passed.

The formerly affected
`physical_transmission_gi_specializations_run_on_hardware_ray_query` control
also passes after localizing the fixture UVs (`20,050` affected pixels,
RGB delta `[28, 19, 247]`).

All clearcoat-normal WGSL feature combinations parse through Naga. The
layered PT implementation remains below the repository's 2,000-line file
limit (`layered_pbr_pt.rs`: 1,958 lines; shared resource helper: 34 lines;
clearcoat-normal module: 328 lines).

## Commits

- `e69b23e` — independent clearcoat-normal transport and qualification;
- `8585425` — shared typed sidecar resource helper and line-policy split;
- `a44bc9b` — real Metal texture-array golden coverage;
- `5c0ab20` — fixture-local UVs preserving unrelated test geometry.

Machine-readable measurements accompany this note in
`docs/evidence/issue-134-clearcoat-normal-pt.json`.
