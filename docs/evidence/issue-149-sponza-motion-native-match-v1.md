# Issue #149 Sponza motion native-match v1 evidence

This checkpoint adds the missing real-scene, full-output-resolution motion
gate for fractional TAA/TSR. The existing governed `sponza-interior` case is a
native-scale still; it cannot qualify the issue #149 requirement that scale
0.75 remain spatially attached to the scene while the camera moves. The exact
comparison base is `75773a7`.

No production shader, renderer resource, or render-graph policy changes in
this checkpoint. It establishes the fail-closed oracle needed to make the next
surface-specific reconstruction change safely.

## Capture contract

`examples/sponza` now accepts an opt-in `--tsr-sequence` mode. It renders a
frame-indexed subpixel camera crawl after a stationary warmup, captures the
current rendered frame synchronously, and exits with an explicit completion
marker. The permanent runner captures the same poses three times:

- native render scale 1.0;
- fractional render scale 0.75;
- an independent fractional 0.75 repeat.

All captures use quality preset 3, TAA enabled, a fixed
`0.016666666667`-second timestep, 16 warmup frames, 32 measured frames, and
pixel-exact 1600x900 headless output. Over the measured interval, camera X
moves from -0.12 to +0.12 and yaw moves from -0.004 to +0.004 radians. The yaw
component is roughly 0.36 output pixels per frame, while the lateral component
exercises real depth-layer disocclusion.

The runner rejects missing or shifted frame numbering, zero-byte captures,
wrong dimensions, absent completion markers, static native controls,
different frame counts, same-run dimension changes, and repeat noise outside
the governed visual reproducibility envelope. It records exact hashes even
when hardware ray queries produce allowed sub-LSB differences.

## Native-reference result

The authoritative Apple M1 Max / Metal capture is stored outside the repository
at `/private/tmp/bloom-sponza-tsr-native-match-v1` (166 MiB). The checked-in
JSON contains the stable aggregate and endpoint hashes.

| Metric, scale 0.75 against native 1.0 | Result | Enforced bound |
|---|---:|---:|
| Normalized RGB frame RMSE | 0.013151819 | <= 0.0133 |
| Normalized RGB motion-derivative RMSE | 0.009808212 | <= 0.0100 |
| Mean luma RMSE | 0.013018815 | recorded discriminator |
| Mean luma SSIM | 0.974230243 | recorded discriminator |
| Mean OKLab delta | 0.009279739 | recorded discriminator |
| Mean edge delta | 0.005236273 | recorded discriminator |

The native negative control changes by 4.086957 mean absolute RGB levels
between adjacent frames; the fractional sequence changes by 3.608425. Thus a
static or duplicated sequence cannot pass by appearing artificially stable.

The worst native-match frame is the first moving frame after the stationary
warmup: luma RMSE 0.014642894, SSIM 0.959338129, OKLab 0.010607342, and edge
delta 0.007181305. By frame 31, the same metrics settle to 0.012769234,
0.975608706, 0.009116412, and 0.004995423. Spatial review localizes the
remaining error to reconstruction detail along the patterned curtain,
stonework, foliage, and geometric edges rather than a returned screen-space
strip. This agrees with the earlier conclusion that another global kernel
change is not justified; the next production experiment needs a reliable
surface/detail discriminator.

## Independent repeat

Metal hardware-ray output is not byte-identical across independent processes,
so the runner uses the same strict visual repeat bounds as the governed quality
manifest instead of weakening or hiding the check.

| Repeat metric across 32 frames | Worst result | Governed bound |
|---|---:|---:|
| Luma RMSE | 0.000188266 | <= 0.002 |
| Luma SSIM | 0.999986172 | >= 0.999 |
| OKLab delta | 0.000001254 | <= 0.001 |
| Edge delta | 0.000001559 | <= 0.001 |

All 32 exact hashes differ, but only 0.000119% of pixels exceed `bloom-diff`'s
default per-pixel tolerance on average. The repeat therefore passes the
governed reproducibility contract with substantial margin.

## Commands

```sh
python3 -m unittest \
  tools.quality.test_sponza_tsr_native_match \
  tools.quality.test_tsr_motion_compare

python3 tools/quality/sponza_tsr_native_match.py \
  --output /private/tmp/bloom-sponza-tsr-native-match-v1 \
  --frames 32 --warmup-frames 16 \
  --width 1600 --height 900 \
  --max-native-frame-rmse 0.0133 \
  --max-native-motion-derivative-rmse 0.0100
```

This is a qualified issue #149 checkpoint, not closure of the parent issue.
The new gate makes Sponza motion a permanent no-regression consumer; the next
step is to isolate SSGI/material contribution and improve the reproducible
moving-detail residual without regressing the synthetic, Bistro, stationary,
or performance gates.
