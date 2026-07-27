# Layered PBR contract

Bloom's layered material work uses one versioned reference model before a lobe
is added to a realtime shader. The checked-in version-1 contract covers the
existing metallic/roughness base layer. Version 2 composes clearcoat and
dielectric specular/IOR over that base. Version 3 adds Charlie sheen and
tangent-space anisotropic GGX. All three are deliberately independent from the
renderer, textures, tone mapping, and scene lighting so energy and reciprocity
failures cannot hide inside an image.

## Version 1 base layer

`tools/bloom-reference/src/layered_pbr.rs` is the authoritative target
evaluator for new layered-material work. The companion
`bloom-brdf-reference` binary uses it to generate the parameter matrix.
Version 1 defines:

- linear base color in `[0, 1]`;
- metallic weight in `[0, 1]`;
- perceptual roughness in `[0.04, 1]`, with `alpha = roughness²`;
- dielectric normal-incidence reflectance `F0 = 0.04`;
- Schlick Fresnel;
- GGX/Trowbridge-Reitz distribution;
- height-correlated Smith visibility including
  `1 / (4 NdotV NdotL)`;
- energy-normalized Burley diffuse;
- reciprocal diffuse interface transmission
  `(1 - F(NdotV)) (1 - F(NdotL))`.

The view/light Fresnel factors matter. Diffuse light enters and leaves a
dielectric interface, while reflected energy remains in the specular lobe.
Using only the half-vector Fresnel term allowed a smooth white dielectric at a
grazing view to retain nearly full diffuse response while its specular
response approached one.

The Burley lobe uses the roughness-dependent Frostbite normalization, fading
from `1` to `1/1.51`. This retains the reciprocal rough-surface response
without the original Disney form's white-furnace gain.

## Version 2 clearcoat and dielectric specular/IOR

Version 2 retains the version-1 result exactly when all new parameters are at
their glTF defaults. When either new lobe is active it defines:

- dielectric `F0 = ((ior - 1) / (ior + 1))²`;
- glTF's `ior = 0` compatibility mode as positive-infinite IOR and `F0 = 1`;
- `KHR_materials_specular` color clamped only after multiplication by the IOR
  reflectance, followed by its scalar factor;
- explicit `F90 = specularFactor`, while pure conductors remain unaffected;
- a max-RGB dielectric diffuse complement so colored specular cannot create
  inverse-colored diffuse energy;
- fixed clearcoat IOR `1.5`, independent perceptual roughness, and the common
  GGX microfacet distribution;
- reciprocal clearcoat transmission at the view and light interfaces,
  attenuating diffuse and base specular before the coat response is added.

The final point is a deliberate energy-conserving refinement of the
non-normative one-sided glTF sample formula. The first implementation used
that one-sided mix and reached approximately `1.252` for a rough conductor
under a coat in a white furnace. Treating the coat as a real interface crossed
on entry and exit removes that unexplained gain while retaining reciprocity.
This convention is the target for every Bloom realtime and path-traced
consumer.

Version 2 does not yet evaluate the independent clearcoat normal map. Import
preserves the normal texture, transform, UV set, and scale losslessly; live
normal sampling is enabled together with the layered shader specialization so
there is no silent scalar approximation in a released rendering path.

## Version 3 sheen and anisotropy

Version 3 remains exactly version 2 when sheen color and anisotropy strength
are zero. Active lobes follow the ratified Khronos conventions:

- sheen uses the Charlie distribution with
  `alphaG = sheenPerceptualRoughness²`;
- visibility uses the full fitted Charlie lambda function, not the older
  Ashikhmin shortcut;
- the base below sheen is scaled by the maximum sheen-color channel and the
  greater of the view/light directional albedos, preserving reciprocity;
- the directional-albedo oracle is a checked 128×128 R16F LUT generated with
  4,096 Charlie-importance samples per texel;
- sheen sits below clearcoat in direct, environment, and physical-transmission
  composition;
- anisotropy uses Burley anisotropic GGX with
  `alphaT = mix(alpha, 1, strength²)` and `alphaB = alpha`;
- visibility is height-correlated anisotropic Smith;
- positive rotation is counter-clockwise in the glTF tangent frame;
- texture RG maps from `[0,1]` to `[-1,1]`, texture B multiplies strength, and
  `KHR_texture_transform` changes lookup coordinates without rotating the
  physical tangent frame.

Valid mesh tangents are authoritative, including `tangent.w` and mirrored
model handedness. If the tangent is absent, zero, or parallel to the normal,
the shader reconstructs a frame from the selected untransformed UV set and
screen-space derivatives. A finite orthogonal fallback covers degenerate UV
derivatives. The realtime roughness floor is `0.04`, matching the established
specular-AA stability contract.

## Qualification corpora

`bloom-brdf-reference` writes a deterministic 48-case sphere parameter matrix:

```shell
cargo run --release \
  --manifest-path tools/bloom-reference/Cargo.toml \
  --bin bloom-brdf-reference -- \
  --out tools/bloom-reference/reference/layered-pbr-v1.json
```

The matrix spans two base colors, dielectric and conductor endpoints, four
roughness values, and three view angles. Each row records separate direct
diffuse/specular values, `BRDF * NdotL`, the current MIS PDF, and white-furnace
directional reflectance. Values are rounded to six decimal places only when
serialized; all tests evaluate full-precision values.

The checked-in JSON is a contract, not an automatically refreshed golden.
Unit tests regenerate it in memory and fail on any drift. Updating it requires
reviewing the formula change and its visual/reference evidence.

White-furnace integration uses deterministic GGX visible-normal importance
sampling for specular and cosine sampling for diffuse. Uniform hemisphere
quadrature is intentionally not used: it aliases the near-delta GGX peak at
the roughness floor and can report false energy gain.

The current gates require:

- finite, non-negative output across the parameter matrix;
- reciprocal BRDF values when view and light are exchanged;
- no more than 2% white-furnace gain at the deliberately conservative
  numerical integration resolution;
- one clamping contract for invalid CPU parameters;
- exact agreement between the generator and checked-in versioned matrix.

The version-2 matrix is generated explicitly:

```shell
cargo run --release \
  --manifest-path tools/bloom-reference/Cargo.toml \
  --bin bloom-brdf-reference -- \
  --version 2 \
  --out tools/bloom-reference/reference/layered-pbr-v2.json
```

It contains 39 cases: 13 base, IOR, specular, clearcoat, and combined
scenarios at three view angles. The checked file's SHA-256 is
`f8b1cdf215b6df2e264b2499b5a2cfaf81da1b18e547ad6aa04af8e21e178497`.
At the deterministic corpus resolution, reflectance ranges from `0.032862` to
`0.967294`. Broader unit sweeps cover metallic `0/0.5/1`, four base
roughnesses, five IOR/specular configurations, four clearcoat
factor/roughness configurations, and four view angles with a `1.02` numerical
tolerance.

The version-3 matrix is generated with:

```shell
cargo run --release \
  --manifest-path tools/bloom-reference/Cargo.toml \
  --bin bloom-brdf-reference -- \
  --version 3 \
  --out tools/bloom-reference/reference/layered-pbr-v3.json
```

Its 30 rows cover ten default, sheen, anisotropy, fabric, coated-fabric, and
all-lobe scenarios at three view angles. The checked JSON SHA-256 is
`0324c95b489f611888163df8ac879033313af6b9190e05af251db91aaca06bfa`.
The checked LUT SHA-256 is
`c4433c235610212432e0da83a0106b8289f08beec9c2524fd82b677945235118`.
Recorded furnace channels range from `0.030422` to `0.834041`; broader tests
also pin reciprocity, finite output, rotation periodicity, the version-2
default, and LUT agreement with a 65,536-sample oracle.

## Defect found by the foundation

The audit found that the existing CPU scene tracer's Smith equation multiplies
the alpha term by
`NdotV`/`NdotL` inside the square root instead of using the squared cosine
form. A rough white conductor at `NdotV = 0.1` returned about `2.06` units in a
unit white furnace when evaluated with that equation. The version-1 target
evaluator uses the correct correlated-GGX equation and pins that grazing case.

The realtime base shader already uses the squared correlated form. The GPU
path-tracing shader and CPU scene tracer still carry their older independent
evaluators. Migrating either is a separate measured slice because the
progressive/reference images change broadly and must be reviewed against an
external model rather than silently overwritten.

## Runtime material-record ABI

The renderer reserves layered-material identity without growing either
existing material record:

- the 176-byte global storage record packs an 8-bit version in bits 24–31 of
  `header.y` and a 24-bit lobe mask in bits 0–23;
- the 80-byte bound/custom uniform stores the exact `u32` version and mask bit
  patterns in `foliage_params.zw` with `bitcast`, not numeric conversion;
- version 1 assigns mask bits 0–5 to clearcoat, sheen, anisotropy,
  iridescence, specular/IOR, and transmission respectively;
- every existing material is version 1 with mask zero.

Version-zero global records are the pre-layered layout. Allocation and update
normalize them to version 1 with an empty mask, so stale data from the old
flags lane cannot activate a future lobe. Tier C still binds the existing ABI
v3 group layout. Foliage setters modify only `foliage_params.xy`, preserving
the metadata lanes.

The WGSL headers expose named version/mask accessors but no production shader
calls them in the ABI foundation. Consequently the base-only path adds no
shader read, branch, bind-group entry, graph pass, image, allocation, or
material-record byte. Quality telemetry reports the two record sizes, both
default masks, the active-lobe material count, and all four zero-cost
invariants.

## Imported material contract

`MaterialLayeredPbr` preserves the complete scalar and source-texture contract
for `KHR_materials_clearcoat`, `KHR_materials_specular`,
`KHR_materials_ior`, `KHR_materials_sheen`, and
`KHR_materials_anisotropy`:

- authored/default identity, factors, clearcoat roughness, clearcoat normal
  scale, specular color, IOR (including zero), sheen color/roughness, and
  anisotropy strength/rotation;
- source texture and image indices plus an optional resolved runtime texture;
- `KHR_texture_transform` offset, rotation, scale, and effective UV set for
  each texture;
- lazy `TEXCOORD_1` retention only when a contributing physical texture
  requests it.

Plain CPU, staged, and runtime glTF/GLB loaders all produce the same material
metadata. Invalid ranges are rejected with the asset material name rather than
clamped silently. The import record is propagated to `PbrMaterial`; it remains
data-only until the corresponding specialized renderer path is present.

## Runtime and compatibility boundary

Clearcoat, specular/IOR, sheen, and anisotropy are live only in lazy scene
specializations. They cover retained and cached opaque geometry, depth-
prepassed opaque geometry, sorted BLEND, TAA-reactive BLEND, weighted OIT, and
combined physical transmission. Scalar, textured UV0/UV1, double-sided,
clustered-light, virtual-shadow, folded scene-input, and native reflection
variants share the same injected source.

All nine layered textures share one sampler. The sheen directional-albedo LUT
is allocated only after the first contributing sheen material and consumes
32,768 persistent bytes. It adds no render-graph pass or transient image.
Base-only materials retain the exact established shader, bind-group layout,
GPU-driven record, and pipeline selection; they do not load or branch on the
layered ABI and do not allocate the LUT.

The CPU scene tracer and GPU path tracer still require an explicit reviewed
migration to these named equations. Iridescence is the next lobe package;
public authoring/debug surfaces and transport parity follow it.
