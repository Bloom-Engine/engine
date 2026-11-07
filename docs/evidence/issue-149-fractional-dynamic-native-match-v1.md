# Issue #149 fractional dynamic native-match v1 evidence

This checkpoint closes the remaining synthetic dynamic-producer coverage gap
for issue #149. The comparison base is
`19ae3947cef0c23ce3e26a74158fdf676640ad9a`. It adds native-matched
fractional-resolution tests for retained rigid, cached skinned, alpha-tested,
physical refraction/reactive, emissive, and procedural foliage paths.

Production shaders, render-graph topology, GPU resources, default pixels, and
default timing are unchanged. The only runtime change is a quality-runner-only
fixed animation clock when `BLOOM_QUALITY_FIXED_TIMESTEP` is set.

## Qualification gap and clock fix

The existing corpus established motion-vector and trail bounds, but object,
skinning, alpha, refraction, emissive, and procedural wind cases did not
independently compare fractional 0.75 reconstruction with native 1.0 TAA.
That allowed a stable but blurred or lagged fractional result to pass.

Procedural foliage also exposed a qualification defect: the governed quality
runner exported `BLOOM_QUALITY_FIXED_TIMESTEP`, but `EngineState` continued to
advance material animation from wall time. A supposedly deterministic capture
could therefore sample a different wind phase on each run.

The engine now consumes a positive finite fixed timestep in quality mode,
caps it at the existing 250 ms maximum delta, resets its deterministic epoch
when changed, and uses it for frame callbacks and renderer material time.
The public `getTime()` API deliberately remains wall elapsed time, so
performance measurement stays truthful. Telemetry reports the timestep that
the engine actually activated rather than echoing the requested value.

## Native-match corpus

Each 256x256 case settles a full 16-phase jitter cycle. Dynamic object cases
then execute four transition and eight recovery frames at native 1.0 and
fractional 0.75 scale. RGB/SSIM similarity, motion-derivative error, and
fresh-epoch recovery are independently bounded. Every negative control makes
a material visual change.

| Case | Movement RGB / outliers | Mean / max RGB | Min SSIM | Derivative | Recovery mean / max outliers | Final RGB |
|---|---:|---:|---:|---:|---:|---:|
| Textured rigid | 13.905919 / 29.4785% | 0.300370 / 0.423436 | 0.984707 | 0.239096 | 0.136122 / 0.3479% | 0.099742 |
| Cached skinned | 14.761546 / 31.7932% | 0.297267 / 0.413584 | 0.985746 | 0.221622 | 0.160990 / 0.4166% | 0.119481 |
| Alpha-tested | 9.597275 / 13.6536% | 0.916391 / 1.103628 | 0.984877 | 0.781872 | 0.462599 / 1.6113% | 0.302424 |
| Refractive/reactive | 9.834137 / 13.5284% | 1.314708 / 1.570429 | 0.956120 | 1.143056 | 0.402529 / 1.8784% | 0.223302 |
| Emissive on | 7.677302 / 6.3522% | 0.390778 / 0.497330 | 0.989451 | 0.161265 | 0.312679 / 0.9003% | 0.215764 |
| Emissive off | 7.721130 / 6.3766% | 0.303009 / 0.389175 | 0.990347 | 0.098195 | 0.238216 / 0.6622% | 0.166870 |

The recovery reference is a separately reset final-state epoch at matched
Halton phases. Every final recovery error is also required to be at most 55%
of its first recovery error, preventing stable path-dependent residue from
passing.

The procedural alpha-tested foliage case renders 16 animated frames. Native
movement is 5.010859 RGB with 11.0550% outliers. Fractional vs native measures
0.760643 mean RGB, 1.127686 maximum frame RGB, 0.988621 minimum SSIM, and
0.938662 derivative error. Repeating the complete fractional sequence from a
fresh fixed-time epoch is byte-for-byte identical.

## Enforced bounds

| Case | Mean / max RGB | Min SSIM | Max derivative | Recovery mean / outliers / final RGB |
|---|---:|---:|---:|---:|
| Textured rigid | <= 0.36 / 0.52 | >= 0.980 | <= 0.30 | <= 0.17 / 0.5% / 0.13 |
| Cached skinned | <= 0.36 / 0.50 | >= 0.980 | <= 0.28 | <= 0.20 / 0.6% / 0.15 |
| Alpha-tested | <= 1.10 / 1.30 | >= 0.980 | <= 0.95 | <= 0.56 / 2.0% / 0.38 |
| Refractive/reactive | <= 1.55 / 1.85 | >= 0.950 | <= 1.35 | <= 0.50 / 2.2% / 0.30 |
| Emissive on | <= 0.48 / 0.60 | >= 0.985 | <= 0.22 | <= 0.40 / 1.2% / 0.28 |
| Emissive off | <= 0.38 / 0.48 | >= 0.986 | <= 0.15 | <= 0.30 / 0.9% / 0.22 |
| Procedural foliage | <= 0.92 / 1.35 | >= 0.985 | <= 1.15 | byte-identical replay |

## Governed runner and performance

The `quick/pbr-spheres-high` governed runner records an active timestep of
0.016666667 and a real 3069.967 ms wall measurement for 300 frames. Its
10.233224 ms wall/frame, 4.541213/9.877041 ms CPU mean/p95, and
15.086233/23.525419 ms GPU mean/p95 prove that the first rejected
implementation—which incorrectly made the public clock report exactly five
seconds—has not survived. The report has only two pre-existing governance
failures: missing `taa-thin-feature-confidence` and an uninstalled approved
baseline.

Four alternating runs of one frozen candidate binary compared normal and
fixed-clock modes directly:

| Run | Wall mean | CPU mean / p95 | GPU mean / p95 |
|---|---:|---:|---:|
| Normal 1 | 11.397594 | 7.115437 / 27.930458 | 31.601292 / 54.799207 |
| Fixed 1 | 11.417869 | 7.047945 / 26.616791 | 32.823155 / 51.540873 |
| Normal 2 | 11.608292 | 7.968421 / 37.687293 | 32.778060 / 52.819587 |
| Fixed 2 | 11.488420 | 7.608012 / 42.857123 | 31.826675 / 51.938419 |

Pairwise wall and GPU-mean deltas change sign. CPU means are lower in both
fixed runs, and GPU p95 is lower in both fixed runs. This is measurement noise,
not a regression. All four final images have the same SHA-256:
`1929adb79316e90085e4ee0c1fd164e2008ed3c673b9b1c59cd99df436b9bf07`.

## Validation

```sh
cargo fmt --manifest-path native/shared/Cargo.toml --all --check
git diff --check
node tools/validate-ffi.js
python3 -m unittest tools/quality/test_run.py -v
cargo check --manifest-path native/web/Cargo.toml --target wasm32-unknown-unknown
cargo test --manifest-path native/shared/Cargo.toml --lib
cargo test --manifest-path native/shared/Cargo.toml --test golden_render -- --nocapture
```

On Apple M1 Max / Metal, the exact tree passes 483 shared-library tests with
one ignored and 86 real-GPU golden tests with three ignored. All 12 quality
runner governance tests pass; FFI validation covers 492 manifest functions on
every platform; the WASM target builds; formatting and diff checks pass.

`node tools/check-file-lines.js` reports six inherited line-ratchet failures in
`post.rs`, `ssgi.rs`, `visibility_buffer.rs`, two virtual-geometry modules, and
`temporal_history.rs`. The exact comparison base reports the same six files and
line counts. This slice adds no ratchet failure; its new module is 878 lines.

The user also accepted the live result as clean. This is a dynamic
reconstruction qualification checkpoint, not closure of issue #149; detailed
Bistro/Sponza coverage, fallback hardware, and the remaining UE-class visual
matrix stay open.
