# Issue #132 bounded spot-light VSM evidence v1

This evidence qualifies explicit spot-light virtual shadows on an Apple M1
Max using Metal and Bloom's native high-end profile.

- Qualified head: `0bcd2c79fa6542aa1894645b9c27b37cfa29cfd4`
- Capture resolution: 2560x1440
- Public API: `addShadowedSpotLight(...)`

## Public and fail-closed contract

The API accepts a position, non-normalized direction, range, full inner and
outer cone angles in degrees, RGB, and intensity. It normalizes the direction
and rejects non-finite values, a zero direction, invalid cone ordering,
non-positive range/intensity, disabled shadows, path tracing, unsupported
tiers, Web, and bounded-list overflow.

Accepted lights enter the shared point-light storage with zero intensity. The
shadow pass restores the authored intensity only after deterministic admission
and after the spot's single perspective page is resident and clean. A rejected,
budget-suppressed, pending, or unsupported spot therefore cannot appear as an
unshadowed direct light.

The shader uses a circular projected-radius test and smooth inner-to-outer
cone attenuation. Equal inner and outer angles take an explicit hard-edge
branch, avoiding undefined equal-edge `smoothstep` behavior. Points outside
the perspective frustum, range, or circular outer cone fail closed.

## Bounded projection and work

The opt-in `quality-stress --vsm-spot-lights` fixture submitted 128
camera-visible shadow-required spots:

| Bound | Observed |
| --- | ---: |
| Public submissions | 128 of 256 |
| Visible requests | 128 |
| Admitted lights | 5 of 5 |
| Budget-suppressed lights | 123 |
| Pages per admitted spot | 1 |
| Local page footprint | 5 |
| Shared residency | 229 of 256 |
| Shared page renders after settling | 0 |

The admitted lights were stable submission indices 0 through 4. Suppression
happened before the retained/immediate light loops and froxel assignment, so
the other 123 requests created neither shading work nor Cartesian page work.

A spot uses one perspective projection, one caster-signature traversal, and
one page instead of a point light's six cube faces. This is an 83.33% page
reduction per admitted local light. It reuses the existing physical depth
array, sampling buffer, page-render pass, metadata lanes, eight-page render
budget, and five-light admission budget. The implementation adds no persistent
GPU bytes and no render-graph pass over the qualified point-light milestone.

## Controlled image oracle

The candidate and reference use the same scene, camera, ambient/environment
terms, 128 spot requests, five admitted spot cones, and all cone/light
parameters. The reference disables caster submission only. This keeps cone
shape and direct-light energy constant and isolates shadow occlusion.

The comparison changed 2.027099609% of pixels at the 0.02 threshold:

- luminance RMSE: 0.016437929;
- SSIM: 0.980780780;
- maximum absolute channel error: 0.470588237;
- mean OKLab delta: 0.001977820;
- mean edge delta: 0.001253707.

The heatmap and side-by-side review localize every material difference to
occluder silhouettes within the authored cones. There are no physical-page
rectangles, projection seams, missing bands, hard cone discontinuities, or
frame-wide lighting/color changes.

SHA-256:

- settled spot VSM frame:
  `9e6378948c3ac2a85646cf68789f15088a5aeec94349aed9c88d54e1f8ef1487`
- same-cone no-caster reference:
  `b9150563d945c890d20a34d248a17e1829937d60b6cd5a160581d21fdbb08e32`
- settled telemetry:
  `1b48c83b5da77901438ca7613df6edbfc051c69f28d6a8bfb40ce1c8bb5d87d7`
- reference telemetry:
  `88e99a53d37f0212e2194b3fdc12d76e4e7f04a6440a7265246595a296b9f6a0`

Both telemetry reports passed
`tools/quality/vsm_local_lights.py --kind spot --min-submitted 100`.

## Compatibility and regression gates

Unsupported raster paths return `false`; they never substitute an unshadowed
spot. Web exposes the same callable ABI as a rejecting stub, and watchOS keeps
manifest parity. The public FFI, package manifest, TypeScript declaration, and
root export all carry the same 13-argument contract.

The complete `scripts/ci-check.sh --quick` lane passed at the qualified head:

- full platform FFI/schema parity and Web arity checks;
- formatting, strict correctness/performance Clippy, and file-size ratchet;
- 356 shared tests passed with 1 ignored;
- the headless negotiated-device test passed;
- 59 GPU goldens passed with 2 hardware-policy tests ignored;
- all 4 render-target tests passed;
- the wasm shared feature check and native Web crate check passed;
- 39 quality-governance/oracle tests passed;
- all visual metric, fault-engine, and asset-cooker tests passed;
- all 20 canonical examples were present.

The ordinary many-point-light immediate and clustered golden tests passed
exactly. The no-local-request uniform early-out remains before local metadata
access, so directional-only rendering does not enter the spot/point branch.

Machine-readable measurements accompany this note in
`docs/evidence/issue-132-spot-lights-v1.json`.
