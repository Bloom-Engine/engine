# Issue #149 moving-detail reconstruction lock v1 evidence

This checkpoint improves fractional TAA/TSR reconstruction during camera
motion without changing settled output, accumulation alpha, render-graph
topology, or bandwidth. The comparison base is
`201b492355eb344cc161b572422ae035922c56d4`.

## Reconstruction contract

The existing RG16Float temporal-provenance G channel now packs an independent
binary detail lock beside history confidence. Unlocked history retains the
legacy 0..1 values byte-for-byte; locked history uses the disjoint 2..3 range,
and reactive history negates the complete payload as before.

Fractional reconstruction seeds the lock only while the camera moves and only
when the already-computed YCoCg luma statistics identify meaningful local
variation. A compatible reprojected lock protects history from transient
source-phase rectification, but does not alter accumulation alpha. Existing
bounds, depth, color-divergence, reactive, and shading rejection clear it, and
a stationary camera clears it immediately. Thus native-scale, half-scale, and
settled fractional output retain their qualified paths.

The capture-only diagnostic is now `taa-detail-lock`: RGB records the current
seed, incoming lock, and validated outgoing lock. It replaces a nine-read
capture-only ridge classifier with the exact production policy.

## Synthetic reference gates

The stationary fractional fixture is exactly unchanged: mean RGB 0.6107025,
RMSE 0.021239817, OKLab 0.002333706, edge error 0.004104466, and SSIM
0.991220445.

| Moving fixture | Metric | Base | Candidate | Enforced bound |
|---|---|---:|---:|---:|
| Glossy slow pan | Mean RGB | 1.105650 | 1.077261 | <= 1.09 |
| Glossy slow pan | Mean SSIM | 0.978452 | 0.978843 | >= 0.9786 |
| Glossy slow pan | Minimum SSIM | 0.973658 | 0.974621 | >= 0.9740 |
| Glossy slow pan | Derivative error | 0.128599 | 0.113418 | <= 0.122 |
| Thin-feature slow pan | Mean RGB | 11.695227 | 10.376356 | <= 11.0 |
| Thin-feature slow pan | Mean SSIM | 0.599727 | 0.720353 | >= 0.66 |
| Thin-feature slow pan | Minimum SSIM | 0.578274 | 0.634745 | >= 0.60 |
| Thin-feature slow pan | Derivative error | 1.193663 | 0.903769 | <= 1.05 |

These gates jointly reject the tempting failure mode where a stale image looks
stable but lags its moving supersampled reference.

## Matched Bistro motion

The real-scene gate renders all 1,176 Bistro placements on Apple M1 Max/Metal
at 512x288. It compares 32 matched camera poses over `(dx=0.08, dz=0.02,
dyaw=0)` at scale 0.75 against the same poses at native scale 1.0. SSGI is off
to isolate reconstruction. Normalized RGB RMSE is evaluated for every frame;
the adjacent-frame RGB derivative is independently compared with native so
history lag cannot masquerade as stability.

| Metric against native | Base | Candidate | Change |
|---|---:|---:|---:|
| Frame RMSE | 0.021735692 | 0.021439823 | -1.36% |
| Motion-derivative RMSE | 0.017314197 | 0.016946073 | -2.13% |

Independent repeat captures measure 0.000247660 candidate-to-candidate RMSE
and 0.000217808 native-to-native RMSE. The committed JSON records endpoint
hashes for the base, candidate, and native sequences. The reusable
`tools/quality/tsr_motion_compare.py` gate rejects incomplete, differently
sized, or regressing sequences.

## Cost and rejected alternatives

The affected TAA pass was measured over 1,200 moving frames at 1600x900,
render scale 0.75. Candidate runs bracketing the exact base measured 1965.765
us and 2092.520 us; the base measured 2042.832 us. The 2029.143 us bracket
mean is 0.67% lower and within combined run noise. The change adds only small
ALU: zero texture reads, passes, targets, bindings, allocations, and persistent
bytes.

Three broader attempts were removed during qualification:

- a continuous axial lock improved images but added 4.9% TAA GPU cost;
- changing accumulation alpha improved fidelity but worsened derivative error;
- allowing the variance lock at rest regressed stationary output.

## Commands

```sh
python3 -m unittest \
  tools.quality.test_tsr_motion_compare \
  tools.quality.test_bistro_temporal_matrix -v

python3 tools/quality/tsr_motion_compare.py \
  --baseline /tmp/bloom-tsr-detail-lock-baseline-v2-20260828 \
  --candidate /tmp/bloom-tsr-detail-lock-candidate-v2-20260828 \
  --native /tmp/bloom-tsr-detail-lock-native-v2-20260828 \
  --expected-frames 32 \
  --output /tmp/issue-149-tsr-motion-compare.json

BLOOM_PROFILE_FRACTIONAL_TAA_FRAMES=1200 \
BLOOM_PROFILE_FRACTIONAL_TAA_RENDER_SCALE=0.75 \
BLOOM_PROFILE_FRACTIONAL_TAA_CAMERA_STEP=0.002 \
cargo test --release --test golden_render \
  quality_presets::profile_fractional_taa_reconstruction -- --exact --nocapture
```

The final tree also passes the complete shared library and real-GPU golden
renderer suites; the exact counts are recorded in the issue checkpoint.
