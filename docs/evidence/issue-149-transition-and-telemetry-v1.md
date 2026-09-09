# Issue #149 transition and reconstruction telemetry v1 evidence

This checkpoint closes two audit gaps around the fractional TAA/TSR path. It
adds an image oracle for projection-only camera transitions and makes runtime
telemetry describe the reconstruction kernels that are actually selected at
each render scale. The comparison base is
`2e9e7440eab4bf01f9cc772c0e5d715701ef5ce3`.

The production shaders and render graph are unchanged by this checkpoint.

## Projection transition oracle

The golden renderer now compares an eight-frame FOV 70 sequence with fresh
history against the same Halton phases immediately after sixteen settled
frames at FOV 42. This directly qualifies the existing policy of remapping a
projection-only jump through motion vectors rather than automatically
invalidating temporal history.

| Frame after FOV step | Mean RGB | Outliers | SSIM |
|---:|---:|---:|---:|
| 0 | 0.0782 | 0.0076% | 0.999575 |
| 1 | 0.0372 | 0.0870% | 0.998096 |
| 2 | 0.1342 | 0.1083% | 0.997061 |
| 3 | 0.1802 | 0.2487% | 0.995744 |
| 4 | 0.1858 | 0.2884% | 0.995776 |
| 5 | 0.0914 | 0.0824% | 0.999031 |
| 6 | 0.0830 | 0.0641% | 0.999017 |
| 7 | 0.0688 | 0.0290% | 0.999124 |

Every frame is gated at mean RGB <= 0.35, outlier fraction <= 1%, and SSIM >=
0.992. Frame seven additionally requires mean RGB <= 0.15 and outlier fraction
<= 0.2%. Explicit camera cuts remain exact after reset; render-scale changes
and resize behavior retain their existing independent coverage.

## Truthful reconstruction telemetry

The settled reconstruction shader selects different source and statistics
kernels by scale. Runtime JSON previously reported the bootstrap Catmull-Rom
path for every settled frame, overstating the 0.75-scale path by five reads.
The telemetry and golden assertions now expose the actual contracts:

| Scale/path | Source reconstruction | Source reads | Additional statistics reads | Composed source reads |
|---|---|---:|---:|---:|
| 0.75 fractional | approximate separable Lanczos2 | 9 | 0 | 9 |
| 0.50 legacy half | approximate radial Lanczos2 | 5 | 4 | 9 |
| native / other | exact separable Catmull-Rom | 9 | 5 | 14 |
| bootstrap | exact separable Catmull-Rom | 9 | 5 | 14 |

At 0.75 scale, the statistics cross reuses five of the nine Lanczos reads. At
0.50 scale, the center source read is reused and four cross reads are added.
The bootstrap contract is reported separately because it intentionally uses
the exact Catmull-Rom path regardless of the settled fractional kernel.

## Static real-scene baselines and rejected broad tuning

Fixed native-reference stills establish a baseline for future targeted work:

| Scene at 0.75 vs native 1.0 | Luma RMSE | SSIM | OKLab | Edge error |
|---|---:|---:|---:|---:|
| Bistro, 512x288 | 0.022507694 | 0.941883624 | 0.010912822 | 0.012089252 |
| Sponza, 1600x900 | 0.016017228 | 0.952767253 | 0.011815320 | 0.007303152 |

Several tempting global changes were removed because they could not improve
all required corpora:

- stronger stationary residual gain regressed absolute native-reference
  error;
- half-strength material mip bias was indistinguishable from repeat noise;
- broader global and depth-adaptive Lanczos kernels regressed untextured or
  geometric references;
- a depth/luma-variance center blend passed generic gates but regressed glossy
  edges and moving fidelity;
- weaker residual gain regressed RMSE, edge error, SSIM, and glossy motion.

This narrows the next quality step: it needs a reliable surface/material
discriminator rather than another global reconstruction-kernel adjustment.

## Validation

```sh
cargo test --test golden_render quality_presets::
cargo test --test golden_render \
  temporal_history::camera_motion_sequence_bounds_ghosting_flicker_and_cut_residue \
  -- --exact
```

Both focused suites pass. On the final exact tree, the complete shared suite
passes 471 tests with one ignored, the real-GPU golden suite passes 78 with
two ignored, and all auxiliary suites pass. The only warning is the existing
unused `mut` in `src/drs.rs`.
