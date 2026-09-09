# Issue #149 moderate-motion phase refresh v1 evidence

This checkpoint targets a temporal-super-resolution failure that remains after
valid detail history has been identified: during moderate camera motion, a
high-frequency fractional-resolution sample can keep accumulating an old
source phase. The result is screen-locked banding or grid motion even though
the underlying surface and history are valid. The exact comparison base is
`b54eba9e1ea65b1b0149e82125a11c64e3d9a676`.

## Production policy

The existing fractional-reconstruction detail classifier is reused. When that
classifier is active and per-pixel motion reaches 0.0005 normalized UV units,
the current-frame contribution receives a 0.15 floor. This affects neither
stationary pixels nor smooth/detail-unlocked pixels. It is applied through the
existing motion blend and therefore retains the established depth, divergence,
reactive, color, and confidence rejection contract.

The change adds no texture reads, bindings, buffers, targets, render passes,
allocations, persistent bytes, or graph topology. It adds one thresholded
selection, multiply, and maximum to the existing full-resolution TAA shader.

## Native-reference motion oracle

The permanent oracle renders two exactly coplanar panels sharing color and a
two-texel high-frequency texture, with perceptual roughness 0.08 and 0.92. A
12-frame camera pan at scale 0.75 is compared with matched 2x raster frames
box-filtered to output resolution. The crop is centered on the moving material
boundary.

| Metric | Base | Candidate | Change | Enforced bound |
|---|---:|---:|---:|---:|
| Mean RGB error | 1.818761 | 1.356988 | -25.39% | <= 1.42 |
| Mean SSIM | 0.974989 | 0.987080 | +0.012091 | >= 0.985 |
| Minimum frame SSIM | 0.942141 | 0.975482 | +0.033341 | >= 0.97 |
| Motion-derivative error | 0.456143 | 0.421135 | -7.67% | <= 0.44 |

The stationary fractional-reference fixture remains byte-for-byte unchanged
at mean RGB 0.610702515 and SSIM 0.991220445. The scale-0.75 glossy slow pan
remains exactly 1.077261 / 0.978843 / 0.974621 / 0.113418 for mean RGB, mean
SSIM, minimum SSIM, and derivative error. The deliberately harsher thin-detail
fixture also remains exactly 10.376356 / 0.720353 / 0.634745 / 0.903769. These
controls show that the gain is isolated to the moderate-motion detail regime,
not a global history reduction or blur trade.

## Rejected history-identity experiments

A stable renderer-authored identity was first packed into the existing
provenance value without adding a resource. Distinct stable keys did not
improve the oracle: lock-only rejection remained at quantization-level parity,
while stronger identity rejection made reference error or temporal variation
worse.

| Policy | Mean RGB | Mean SSIM | Minimum SSIM | Derivative error | Decision |
|---|---:|---:|---:|---:|---|
| Base | 1.818761 | 0.974989 | 0.942141 | 0.456143 | accepted baseline |
| Stable-key lock kill only | 1.818452 | 0.974988 | 0.942147 | 0.456155 | neutral; removed |
| Full stable-key rejection | 1.824365 | 0.974837 | 0.942015 | 0.489838 | regression; removed |
| Nearest matched history | 1.828624 | 0.974586 | 0.943273 | 0.529244 | regression; removed |
| Same-surface bilinear taps | 1.832525 | 0.974367 | 0.940711 | 0.486263 | regression; removed |

This falsifies cross-material history as the dominant error in this fixture.
The gain instead comes from bounding valid but source-phase-lagged history.

## Performance

Exact frozen release binaries were measured on Apple M1 Max / Metal at
1600x900 output, scale 0.75, with 1,200 moving frames per run and camera step
0.002. Three baseline runs were bracketed by four candidate runs.

| Run | TAA GPU time (us) |
|---|---:|
| Candidate 1 | 1826.640 |
| Baseline 1 | 2009.076 |
| Candidate 2 | 1603.400 |
| Baseline 2 | 1461.669 |
| Candidate 3 | 1873.366 |
| Baseline 3 | 1986.851 |
| Candidate 4 | 1809.632 |

The mean of the three candidate brackets is 1764.967 us versus 1819.199 us
for their baseline centers (-2.98%). Individual paired deltas change sign and
are larger than the aggregate difference, so this is correctly classified as
no measurable regression rather than a performance improvement.

An explicit ignored `profile_fractional_taa_reconstruction` test now owns the
profiling entry point documented by earlier evidence instead of relying on an
environment-controlled side effect inside a visual-quality test.

## Validation state

- all 11 quality-preset visual tests pass;
- the dedicated release performance gate passes;
- shared library suite: 482 passed, one ignored;
- final exact-tree real-GPU corpus: 79 passed, three ignored;
- formatting and diff whitespace checks pass;
- exact candidate Bistro build loads all 2,909 placements with hardware ray
  query SSGI; the interactive cobble, floor-grid, bright-facade, rail/wire, and
  coplanar-detail motion sweep was reported clean.

## Commands

```sh
cargo test --lib
cargo test --test golden_render quality_presets:: -- --nocapture
cargo test --test golden_render -- --nocapture

BLOOM_PROFILE_FRACTIONAL_TAA_FRAMES=1200 \
BLOOM_PROFILE_FRACTIONAL_TAA_RENDER_SCALE=0.75 \
BLOOM_PROFILE_FRACTIONAL_TAA_CAMERA_STEP=0.002 \
cargo test --release --test golden_render \
  quality_presets::profile_fractional_taa_reconstruction \
  -- --exact --ignored --nocapture
```

This is a qualified issue #149 slice, not closure of the parent issue. The
remaining native-matched corpus, fast transition recovery, platform matrix,
and broader UE-class reconstruction acceptance criteria remain open.
