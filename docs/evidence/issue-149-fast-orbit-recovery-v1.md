# Issue #149 fast-orbit recovery v1 evidence

This corpus-only checkpoint adds the missing native-matched fast camera-motion
oracle for issue #149. The comparison base is
`ae2e6b7fe5b3ff1cf0e8cbccda7a7ce2cad4cd27`. Production shaders, resources,
render-graph topology, memory, and pixels are unchanged.

## Gap closed by this oracle

The existing temporal corpus already proves that an explicit camera cut
produces an exact fresh epoch, a 42-to-70-degree FOV step stays close to a
phase-matched fresh history, and a large pose jump sheds severe trails within
four frames. Its fast-rotation gate, however, compared fractional output only
with its own later settled estimate. A blurred or otherwise non-native
reconstruction could therefore pass.

The new 256x256 real-GPU fixture independently renders identical camera paths
with native 1.0 and fractional 0.75 TAA. The camera orbits a high-frequency
grid, 33 alternating thin columns, and strongly colored primitives. It moves
8.130852 metres and rotates 1.2 radians over four frames, then holds the final
pose for eight recovery frames. Sixteen old-pose frames align the 16-phase
jitter cycle before motion.

The fractional recovery frames are also compared with a separately reset
final-pose epoch at the same Halton phases. That comparison prevents a stable
but path-dependent old image from passing as clean recovery.

## Native-match gates

The native negative control changes by mean RGB 23.392517 with 29.4907% of
pixels over the existing outlier threshold, so the fixture exercises a
material transition rather than subpixel noise.

| Metric, fractional 0.75 vs native 1.0 | Measured | Enforced bound |
|---|---:|---:|
| Mean RGB across 12 frames | 1.062483 | <= 1.12 |
| Maximum frame mean RGB | 1.271566 | <= 1.34 |
| Minimum frame SSIM | 0.966792 | >= 0.962 |
| Mean motion-derivative error | 1.094460 | <= 1.16 |

RGB/SSIM and derivative error are gated independently. Lower temporal
variation cannot compensate for an image that lags the moving native result.

## Fresh-epoch recovery gates

| Recovery frame | Mean RGB | Outlier fraction | SSIM |
|---:|---:|---:|---:|
| 0 | 0.435740 | 0.5402% | 0.990881 |
| 1 | 0.273702 | 0.2609% | 0.995364 |
| 2 | 0.199092 | 0.1099% | 0.997844 |
| 3 | 0.154073 | 0.0381% | 0.998637 |
| 4 | 0.134120 | 0.0168% | 0.999116 |
| 5 | 0.115234 | 0.0061% | 0.999320 |
| 6 | 0.104411 | 0.0046% | 0.999428 |
| 7 | 0.092255 | 0.0046% | 0.999525 |

The eight-frame recovery mean is 0.188578 RGB and its maximum outlier fraction
is 0.5402%; these are gated at 0.20 and 0.6%. The final frame must reach mean
RGB <= 0.11, SSIM >= 0.9994, and at most 25% of the first recovery-frame error.

## Validation

```sh
cargo test --lib
cargo test --test golden_render \
  temporal_history::fractional_fast_orbit_tracks_native_motion_and_fresh_recovery \
  -- --exact --nocapture
cargo test --test golden_render -- --nocapture
cargo fmt --check
git diff --check
```

On Apple M1 Max / Metal, the exact tree passes 482 shared-library tests with
one ignored and 80 real-GPU golden tests with three ignored. Formatting and
diff whitespace checks pass. Because this checkpoint adds only a test and
evidence, performance and runtime resource contracts are exactly unchanged.

This closes a synthetic fast-orbit corpus gap, not issue #149. Full detailed
Bistro/Sponza fast paths, object/foliage motion, fallback platforms, and the
remaining UE-class acceptance matrix stay open.
