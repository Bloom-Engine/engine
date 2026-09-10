# Portable cutout coverage phase

Sponza and skinned/alpha motion now pass the existing portable image thresholds
on Radeon 760M/Vulkan using the modern tier and software GI. The correction makes
the approved material-space coverage phase explicit with integer arithmetic.
Approved images, quality thresholds, and sampled alpha textures are unchanged.

## First divergent stage

Exact attachment captures at diagnostic source `30e7625` expose the cutout
decision on the nearest opaque card, including rejected fragments. Separate U
and V captures preserve each coordinate's complete f32 bit pattern in the albedo
MRT. Both channels have identical depth, HDR inputs, and decision flags on each
backend. On Windows, all 51,820 accepted nearest-card fragments match the
unmodified canonical scene's depth within `1e-6`. These probes change occlusion
and therefore describe the nearest card, not every layer of the original leaf.

Across 103,644 corresponding card pixels, most UV differences are tiny: the
median absolute difference is about `1.49e-8` in each coordinate. Despite that,
69,212 Bayer thresholds and 11,131 survival decisions differ. Only 267 survival
disagreements occur where the thresholds agree. Coplanar geometry can select
different triangles at the same depth, so large UV outliers are retained.

The shader computed its phase extent with:

```wgsl
floor(vec2<f32>(dimensions) * exp2(-max(floor(lod), 1.0)))
```

On the tested Radeon, powers of two produced the full mip dimension. On hosted
Metal, the observed phase extent was one smaller for the power-of-two foliage
texture. The smaller extent predicts all 101,403 inspected Metal thresholds
away from integer LOD boundaries; the full extent disagrees at 67,103 of those
pixels. Conversely, the full extent predicts all 101,419 inspected Windows
thresholds. The exclusions account for half-precision LOD in the HDR capture;
UVs retain full precision.

WGSL permits error in `exp2`, so an approximate result immediately below an
integer boundary can change the subsequent floor. The old expression required
more numerical precision than its operation guarantees. This observation does
not establish a driver defect. See the [WGSL floating-point accuracy contract](https://www.w3.org/TR/2026/CRD-WGSL-20260831/#floating-point-accuracy).

## Correction

Scene and shadow shaders now derive the repeating phase extent from the final
source texel index using integer arithmetic:

```text
level = clamp(floor(lod), 1, 31)
phase_extent = max((texture_dimensions - 1) >> level, 1)
```

This preserves the grid observed in the approved Metal captures and makes it
identical across backends. Reducing the final source texel index also handles
odd texture dimensions: subtracting one from the already reduced mip count
would change those cases incorrectly. The phase calculation leaves the alpha
sample's coordinates, filtering, mip selection, and coverage probability intact.
The level bound and minimum extent handle very coarse LODs and one-texel axes.

## Local image results

The same frozen example commands, modern capability tier, software GI, 120
warm-up frames, 240 measured frames, and raw attachment export produced:

| Scene | Previous SSIM | Corrected SSIM | Corrected luminance RMSE | Image gate |
| --- | ---: | ---: | ---: | --- |
| Sponza | 0.969274342 | 0.986160457 | 0.009024118 | Pass |
| Skinned/alpha motion | 0.931530774 | 0.990073442 | 0.017359400 | Pass |

These focused captures establish image behavior, not hardware timing budgets.
Their executable hashes, commands, raw attachments, metrics, and logs are retained
under `tools/quality/out/windows-engine-plan/alpha-phase-fix/`.

The GPU regression executes the production scene and shadow threshold functions
against twelve fixed reference points. It covers power-of-two and odd extents,
UV repetition, fractional LOD, one-texel axes, and coarse LODs. The corrected
Radeon/Vulkan and DX12 paths pass all 24 values. The old Vulkan shader fails the
first reference point (`0.03125` instead of `0.34375`). Local contracts, strict
lint, and the complete shared suite pass, including 91 reported golden-test
passes with four ignored tests; optional external-input fixtures remain separate
acceptance requirements. Full-corpus, repeated-run, and hosted validation are
recorded separately as they complete. This change does not close
the remaining temporal, packaging, platform, or named RTX 4080 requirements in
the [engine completion plan](../windows-engine-plan.md).
