# Issue #27 MASK/cutout compatibility decision v1

This checkpoint fixes the first visibility-buffer activation contract at
revision `f6518405a13b615d6f7708534f4b02887e3d5b02`: glTF MASK/cutout geometry
remains on Bloom's alpha-aware forward compatibility path. It is not admitted
to visibility shading in version 1.

## Decision

Bloom's existing depth prepass already evaluates cutout alpha and primes the
winning depth before the forward MRT pass. Moving MASK into visibility shading
would require the ID raster and reconstructed PBR path to own the exact texture,
sampler, UV transform, cutoff, coverage-mip, derivative/LOD, and ordering
contract. It would not remove hidden PBR shading that the current prepass
already rejects. The measured visibility path has also not demonstrated a
general performance win, so accepting silhouette or ordering risk for this
material class has no supporting evidence.

The V1 contract is therefore explicit:

- static fully opaque Tier-A draws may enter visibility shading;
- MASK/cutout draws keep the established forward alpha/depth owner;
- the visibility result remains the background layer at discarded mask texels;
- surviving mask texels replace visibility HDR and every dependent MRT at
  their ordinary depth/order;
- capability telemetry counts the MASK draw as compatibility geometry.

An alpha-aware visibility path remains possible later, but it requires a new
independent image, ordering, derivative/coverage, and total-GPU qualification.
It is not an implicit extension of the opaque path.

## Real-GPU oracle

The process-isolated `visibility_buffer_parity` test renders the identical
160x128 scene in forward/off and visibility/shade processes. Thirty-two moving
opaque draws are visibility-eligible. Six moving layered opaque draws and one
moving MASK draw are forward compatibility-owned. The MASK triangle overlaps
an eligible final-row triangle and uses a 2x2 texture with two opaque and two
transparent texels, so both ownership branches are visible in the same draw.

The test now requires at least seven compatibility draws in runtime telemetry;
silently admitting the MASK draw makes the gate fail. The clean-revision Apple
M1 Max/Metal run passed with these output deltas:

| Output | Changed components | Maximum delta | Mean delta |
|---|---:|---:|---:|
| Final RGBA8 | 187 / 81,920 | 1 code | 0.00228271 code |
| HDR RGBA16F | 1,747 / 81,920 | 0.000854492 | 0.000004231930 |
| Material RG8 | 36 / 40,960 | 1 code | 0.000878906 code |
| Velocity RG16F | 55 / 40,960 | 0.000003815 | 0.000000005122 |
| Albedo RGBA8 | 76 / 81,920 | 1 code | 0.000927734 code |

The reference velocity attachment contains more than 256 non-zero moving
components, preventing a static/all-zero pass. All four MRTs are finite, the
final output stays within one LSB, and the full two-test release target passes.

## Scope

This closes the #27 decision about MASK/cutout ownership. It does not activate
visibility shading by default or close the broader Bistro/effects, performance,
or discrete-plus-integrated hardware gates.
