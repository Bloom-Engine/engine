# Issue #149 detail-locked moving residual v1 evidence

This checkpoint fixes the first reproducible material-specific defect exposed
by the governed full-resolution Sponza motion gate: fractional reconstruction
loses authored microtexture and alpha-cutout edge energy during camera motion.
The exact comparison base is `08d0010`.

## Owner isolation

The identical 32-frame, 1600x900 native/fractional/repeat matrix was captured
with SSGI enabled and disabled before changing production rendering.

| Metric | SSGI on | SSGI off | Difference |
|---|---:|---:|---:|
| Native frame RMSE | 0.013151819 | 0.013158897 | +0.000007078 |
| Native motion-derivative RMSE | 0.009808212 | 0.009800298 | -0.000007914 |
| Mean luma RMSE | 0.013018815 | 0.013027297 | +0.000008482 |
| Mean SSIM | 0.974230243 | 0.974279135 | +0.000048893 |

The changes are approximately five hundredths of one percent in the two
reference errors. SSGI is therefore not the owner. A fixed frame-31 spatial
audit then localized the largest mismatch to alpha-cutout foliage (SSIM
0.922515, edge delta 0.010678), with smaller but coherent loss in the patterned
curtain, stone, and smooth painted ceiling.

## Production policy

The existing fractional Lanczos statistics already classify a moving pixel as
authored high-frequency detail when its reconstructed luma standard deviation
exceeds both a relative and absolute floor. That classifier seeds the existing
persistent detail lock.

The accepted change retains the qualified 2% moving reconstruction residual
on every fractional surface and raises it to 6% only while that exact detail
lock is active. Smooth surfaces retain the previous path. The policy does not
change accumulation alpha, history lifetime, rejection, neighborhood bounds,
sampling kernels, or render-graph topology. It adds one classifier-weighted
multiply/add and adds zero texture reads, samplers, bindings, buffers, targets,
passes, allocations, or persistent bytes.

Quality telemetry now reports:

- base moving detail strength `0.02`;
- policy `detail-lock-weighted`;
- locked moving detail strength `0.06`;
- classifier `fractional-luma-variance-lock`;
- zero additional samples.

## Full-resolution Sponza native match

| Metric, scale 0.75 against native 1.0 | Base | Candidate | Change | New gate |
|---|---:|---:|---:|---:|
| RGB frame RMSE | 0.013151819 | 0.013108898 | -0.33% | <= 0.01313 |
| RGB motion-derivative RMSE | 0.009808212 | 0.009744127 | -0.65% | <= 0.00978 |
| Mean luma RMSE | 0.013018815 | 0.012986683 | -0.25% | <= 0.0130 |
| Mean SSIM | 0.974230243 | 0.974700231 | +0.000470 | >= 0.9745 |
| Mean OKLab delta | 0.009279739 | 0.009220911 | -0.63% | <= 0.00925 |
| Mean edge delta | 0.005236273 | 0.005114727 | -2.32% | <= 0.0052 |

The stricter gate set rejects the exact base result and accepts the candidate.
Independent candidate repeats remain far inside the governed hardware-ray
noise envelope: worst luma RMSE 0.000131256, minimum SSIM 0.999996364,
worst OKLab 0.000001625, and worst edge delta 0.000001706.

The first moving frame remains the hardest transition, but all perceptual
metrics improve. By frame 31, candidate luma RMSE is 0.012733491, SSIM is
0.976132929, OKLab is 0.009048228, and edge delta is 0.004858705.

## Material-region discriminator

The fixed frame-31 regions are diagnostic discriminators rather than a second
baseline installation. Every region improves against its independently
captured matched native frame.

| Region | Metric | Base | Candidate | Change |
|---|---|---:|---:|---:|
| Patterned curtain | SSIM | 0.969739 | 0.970941 | +0.001202 |
| Patterned curtain | Edge delta | 0.007166 | 0.006875 | -4.05% |
| Stone vault | SSIM | 0.982596 | 0.982864 | +0.000268 |
| Stone vault | Edge delta | 0.003264 | 0.003206 | -1.76% |
| Alpha-cutout foliage | SSIM | 0.922515 | 0.922938 | +0.000423 |
| Alpha-cutout foliage | Edge delta | 0.010678 | 0.010574 | -0.97% |
| Painted ceiling | SSIM | 0.982633 | 0.983072 | +0.000438 |
| Painted ceiling | Edge delta | 0.003539 | 0.003449 | -2.53% |

This is a bounded spatial gain rather than global sharpening: the smooth
ceiling remains close to native while the strongest improvement occurs on the
detail-classified curtain, and no region regresses.

## Synthetic and dynamic controls

All focused real-GPU quality-preset tests pass. The synthetic references
confirm that the same policy helps multiple detail classes while retaining the
established derivative bounds.

| Fixture | Metric | Base | Candidate |
|---|---|---:|---:|
| Glossy slow pan | Mean RGB | 1.077261 | 1.074287 |
| Glossy slow pan | Mean SSIM | 0.978843 | 0.979577 |
| Glossy slow pan | Minimum SSIM | 0.974621 | 0.975349 |
| Glossy slow pan | Derivative error | 0.113418 | 0.118591 |
| Thin-feature slow pan | Mean RGB | 10.376356 | 10.052485 |
| Thin-feature slow pan | Mean SSIM | 0.720353 | 0.742975 |
| Thin-feature slow pan | Minimum SSIM | 0.634745 | 0.640565 |
| Thin-feature slow pan | Derivative error | 0.903769 | 0.981522 |
| Coplanar material boundary | Mean RGB | 1.356988 | 1.350689 |
| Coplanar material boundary | Mean SSIM | 0.987080 | 0.987222 |
| Coplanar material boundary | Minimum SSIM | 0.975482 | 0.975973 |
| Coplanar material boundary | Derivative error | 0.421135 | 0.415797 |

The glossy and thin-feature derivative values remain below their established
0.122 and 1.05 hard bounds. The complete dynamic corpus also passes rigid,
skinned, alpha-tested, procedural-foliage, refraction/reactive, and emissive
native-match and recovery cases.

## Performance and validation

Exact release binaries at base and candidate were alternated for three
1,200-frame profiles at 1600x900 output, scale 0.75, camera step 0.002.

| Revision | TAA GPU runs (us) | Mean |
|---|---|---:|
| Base `08d0010` | 1497.615, 1758.967, 1506.277 | 1587.620 |
| Candidate | 1593.861, 1522.042, 1562.422 | 1559.442 |

The candidate mean is 1.78% lower. Individual pairs change sign and their
ranges overlap substantially, so this is classified as no measurable
regression, not a performance improvement.

Final validation:

- all 69 quality-tool unit tests pass;
- focused quality-preset GPU tests: 11 passed, one profiling test ignored;
- complete shared library: 483 passed, one ignored;
- complete real-GPU renderer corpus: 86 passed, three ignored;
- explicit release performance profile passes;
- Sponza example builds with all 492 FFI functions;
- Rust formatting and diff whitespace checks pass.

The only warning is the pre-existing unused `mut` in `src/drs.rs`.

## Commands

```sh
python3 tools/quality/sponza_tsr_native_match.py \
  --output /private/tmp/bloom-sponza-tsr-detail-residual-candidate-v1 \
  --frames 32 --warmup-frames 16 --width 1600 --height 900 \
  --max-native-frame-rmse 0.01313 \
  --max-native-motion-derivative-rmse 0.00978 \
  --max-mean-luma-rmse 0.0130 \
  --min-mean-ssim 0.9745 \
  --max-mean-oklab-delta 0.00925 \
  --max-mean-edge-delta 0.0052

cargo test --lib
cargo test --test golden_render -- --nocapture

BLOOM_PROFILE_FRACTIONAL_TAA_FRAMES=1200 \
BLOOM_PROFILE_FRACTIONAL_TAA_RENDER_SCALE=0.75 \
BLOOM_PROFILE_FRACTIONAL_TAA_CAMERA_STEP=0.002 \
cargo test --release --test golden_render \
  quality_presets::profile_fractional_taa_reconstruction \
  -- --exact --ignored --nocapture
```

This is a qualified issue #149 quality checkpoint, not closure of the parent
issue. It fixes the first Sponza material-specific moving-detail residual while
preserving the broader temporal, dynamic, resource, and performance contracts.
