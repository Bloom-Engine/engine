# Issue #149 coplanar material-boundary v1 evidence

This checkpoint adds a permanent native-reference motion oracle for a failure
that depth provenance cannot identify: two exactly coplanar surfaces sharing
the same base color and high-frequency texture but using glossy and rough
material responses. Production rendering remains unchanged. The comparison
base is `e12f3d4053dd7a07ff067cae9cf4fc4c1312c0e2`.

## Oracle

The fixture renders two adjacent quads at one depth with perceptual roughness
0.08 and 0.92. Both use the same two-texel checker and base color. A 12-frame
camera pan at 0.75 render scale is compared with matched 2x raster captures
box-filtered to output resolution. Metrics are evaluated in a 32x160 crop
centered on the moving material boundary so the background cannot dilute the
result.

| Metric | Accepted baseline | Enforced bound |
|---|---:|---:|
| Mean RGB error | 1.818761 | <= 1.86 |
| Mean SSIM | 0.974989 | >= 0.9745 |
| Minimum frame SSIM | 0.942141 | >= 0.94 |
| Motion-derivative error | 0.456143 | <= 0.47 |

The RGB/SSIM bounds prevent same-depth cross-material history from drifting
away from native. The independent derivative bound prevents a superficially
clean still from passing by retaining stale material detail.

## Rejected compact material-class experiment

A production prototype sampled the existing current-frame RG8 material
surface once per TAA output pixel, quantized roughness into three broad
classes, and packed that class beside confidence/detail/reactive state in the
existing RG16F provenance channel. Half-float spacing remained exact and no
new history image was required, but the path added one full-screen texture
read and one binding.

Three policies were evaluated and removed:

| Policy / fixture | Mean RGB base -> candidate | SSIM base -> candidate | Derivative base -> candidate |
|---|---:|---:|---:|
| Full mismatch rejection, gentle uniform-panel pan | 0.148244 -> 0.146714 | 0.999155 -> 0.999161 | 0.017981 -> 0.018478 |
| 50% mismatch rejection, faster narrow-boundary pan | 0.217459 -> 0.214269 | 0.998074 -> 0.998065 | 0.050829 -> 0.051781 |
| Rectification-lock kill only, textured oracle | 1.818761 -> 1.818452 | 0.974989 -> 0.974988 | 0.456143 -> 0.456155 |

Strict and partial rejection improved RGB error slightly but made temporal
variation worse. Restricting the class to the moving detail lock was
effectively neutral. None justified permanent full-screen bandwidth, so all
shader, binding, packing, and telemetry changes were removed. This also avoids
repeating the earlier rejected packed-normal experiment recorded on issue
#149.

The result narrows future work: a useful discriminator must come from a stable
renderer-authored identity or reuse data already fetched by TAA. Reclassifying
sampled material properties at resolve time is not sufficient.

## Validation

```sh
cargo test --test golden_render \
  quality_presets::fractional_coplanar_material_boundary_tracks_supersampled_motion \
  -- --exact --nocapture
cargo test --test golden_render quality_presets::
```

The new oracle and all eleven quality-preset tests pass on Apple M1 Max / Metal.
On the final exact tree, the complete shared suite passes 471 tests with one
ignored, the expanded real-GPU golden suite passes 79 with two ignored, and
all auxiliary suites pass. The only warning is the existing unused `mut` in
`src/drs.rs`.
