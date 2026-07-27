# Transparent global illumination

Bloom's screen-probe global illumination represents imported physical
transmission with one bounded colored continuation. Glass remains visible to
the normal first-hit query, but radiance behind the nearest glass surface is no
longer discarded or replaced with an opaque card.

## Activation and zero-cost contract

The route activates only when all of the following are true:

1. screen-probe GI is enabled;
2. imported physical transmission is enabled;
3. at least one visible retained scene node or GI-only proxy has both a
   Mesh-Card slot and active physical transmission; and
4. `BLOOM_TRANSPARENT_GI` is not set to `0`, `false`, `off`, or `disabled`.

Scene preparation maintains an O(1) transmission gate. An ordinary opaque
scene does not read the kill switch, scan nodes for transmission, create or
select a specialized pipeline, change its TLAS masks, or execute a
transparent-GI shader branch. The ordinary shaders keep a compile-time-false
switch and preserve their established first-hit code.

Activation adds no texture, buffer, bind group, graph resource, graph pass, or
transient slot. It reuses spare lanes in the existing per-instance GI record.
The exact additional persistent and transient allocation is therefore zero
bytes. Three specialized pipelines are compiled lazily: hardware probe trace,
software SDF probe trace, and hardware world-space radiance-cache bake.

## Hardware ray-query representation

Ordinary instances retain the `0xff` visibility mask. When the route is
eligible, a physical-transmission instance uses bit 1 (`0x02`) while opaque
instances retain bit 0. The normal `0xff` ray query therefore still observes
the nearest surface of either kind.

Only when that first hit is glass does the specialized kernel issue one
additional query with mask `0x01`. That query skips the complete glass
instance, including its back face, and finds the nearest opaque surface behind
it. Probe tracing and hardware world-space radiance-cache baking share this
rule. The number of continuation queries is capped at one regardless of the
number of overlapping transparent instances.

## Software SDF representation

When the software clipmap backend is active, physical-transmission meshes are
excluded from the scene-wide opaque SDF. The established SDF trace therefore
continues to find the nearest opaque surface behind glass. The specialized
kernel separately intersects the ray with retained instance metadata and keeps
only the nearest conservative transmission world AABB before that opaque hit.

This preserves the existing clipmap resolution, bake schedule, storage, and
probe dispatch dimensions. A route change invalidates an in-flight or live
clipmap, and a staging bake is published only if its scene version and
transmission mode are still current.

## Transport model

The existing 144-byte `InstanceGiDataCpu` record carries:

| Lane | Value |
| --- | --- |
| `card_aabb_min.w` | Beer-Lambert absorption red |
| `card_aabb_max.w` | Beer-Lambert absorption green |
| `world_aabb_min.w` | Beer-Lambert absorption blue |
| `world_aabb_max.w` | BLEND coverage |
| `mat_params.z` | scalar transmission × non-metallic weight |
| `mat_params.w` | dielectric Fresnel pass fraction |

Absorption uses authored attenuation color and distance, with thickness scaled
by average world model scale. The bounded composition is:

```text
front_radiance * coverage * (1 - transmission_weight)
  + behind_radiance
    * mix(
        1,
        base_color
          * absorption
          * transmission_weight
          * fresnel_pass,
        coverage
      )
```

The `mix` preserves the uncovered part of a BLEND surface while tinting only
its covered fraction. The formula avoids adding the opaque front-card result
on top of fully transmitted radiance and conserves the scalar
surface/transmission partition. Fully metallic authored transmission remains
in the opaque TLAS/SDF subset because its dielectric transport weight is zero.

Changing physical-transmission metadata invalidates the per-instance data,
TLAS mask, and software SDF membership exactly once. Switching the route also
forces one probe-history refresh and invalidates world-space radiance-cache
cascades so the old visibility representation cannot linger temporally.

## Explicit limits

- At most one nearest transmission instance contributes to a GI ray.
  Arbitrarily deep dielectric stacks are outside this bounded baseline.
- The software fallback uses the instance world AABB, not triangle-accurate
  entry depth, for selecting its nearest glass layer.
- GI transport currently uses scalar transmission, thickness, and metallic
  factors plus the instance's flat base albedo. Per-hit transmission,
  thickness, metallic-roughness, and base-color texture modulation remains
  camera/shadow shading work and is not sampled by the continuation query.
- The route follows Bloom's established scene-GI contract: retained nodes and
  GI-only proxies enter Mesh-Cards, TLAS, and SDF data. Immediate cached model
  draws do not.
- Dynamic skinned path-tracing instances retain the existing opaque GI
  metadata until skinned retained-scene GI proxies are introduced.

## Diagnostics and qualification

Native quality telemetry under `renderer_paths.transparent_gi` exposes:

- `enabled` and `active`;
- `representation` (`one-layer-colored-continuation`);
- `additional_persistent_bytes` (`0`); and
- retained `instance_count`.

Live GPU property tests execute both hardware ray-query and software SDF
specializations. The hardware test changes only a GI-only slab from opaque to
physical transmission and verifies that a cyan/green emitter behind it changes
the indirect result while camera-facing geometry remains identical. Shader
tests parse all ordinary and specialized variants and assert the bounded
transport math.

A sustained Metal A/B used 96 moving retained physical-transmission panes at
960×540, with 120 warm-up and 240 measured frames. Transmitted shadows were
disabled per node so the comparison isolated transparent GI. The specialized
hardware probe trace averaged 0.254 ms versus 0.195 ms for the opaque control,
an added 0.059 ms in this deliberately saturated route. Feature-on total GPU
p95 was lower within run-to-run noise (90.88 ms versus 93.41 ms), as was CPU
p95 (8.38 ms versus 9.19 ms). Both routes compiled one identical 24-pass graph,
used the same three transient slots and 26,956,800 transient bytes, and
produced a bit-identical camera image in the neutral stress scene. The
separate emitter property test supplies the non-neutral visual assertion.
