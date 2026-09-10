# Windows/Vulkan stationary software SSGI

The Radeon 760M exposed phase-dependent flicker in a stationary scene even though
scene color, depth, geometry, and the complete 16-phase radiance history were
unchanged. The first changing stage was the probe temporal/spatial accumulator.
A current eight-ray neighborhood repeatedly clamped the converged history and
fed that change into the following frame.

The fix recognizes repeated software phases on the same world surface. Once all
16 phases repeat, the renderer uses the complete integral without the noisy
current-neighborhood clamp. Any changed phase revokes that state; geometry
reprojection and the existing responsive blend/clamp remain in effect during
changes. Current software integrals are explicitly rounded to their retained
RGBA16F precision before comparison and reduction, so texture-store rounding
cannot keep an otherwise identical phase from matching. The hardware cache's
coherence rules are preserved. No texture, buffer, pass, ray count, golden,
threshold, or existing performance budget is added or increased.

## Correctness evidence

Adapter: AMD Radeon 760M Graphics, Vulkan, integrated GPU, AMD 24.12.1 (LLPC),
Windows 11. The renderer fix is commit
`5499bc1057a726397f53aebc40c885f06a0d64f5`. The frozen comparison uses its shader
changes over `3767b23`; executable and shader hashes are retained in
`tools/quality/out/windows-engine-plan/ssgi-isolation/frozen/receipt.json`.

| Static output | Previous shader SSIM | Fixed shader SSIM |
| --- | ---: | ---: |
| 256 x 256 | 0.998438092 | 1.0 |
| 1280 x 720 | 0.982902534 | 1.0 |

The five independent runs at each size reproduce these image metrics. Raw ray
realizations still advance. In 34 raw frame captures, all 228 valid probes have
stationary integrals and reconstructed output; the 28 invalid/sky probes remain
invalid. Full-precision header and RGBA16F trace/history dumps retain the first
stage evidence.

A new regression exercises bright, dim, and bright-again lighting without a
history reset between states. Each state's complete 16-phase output cycle is
byte-identical after 32 frames, and the returning bright state matches a fresh
64-frame reference exactly. Dimming changes mean RGB by 5.31909, ruling out
frozen or disabled GI. All captures contain finite HDR radiance. The previous
shader fails this same regression at the first stationary phase.

The final golden batch passes 89 tests, with 4 explicitly ignored and 2 optional
cases returning for absent external Bistro inputs. There are no unavailable-GPU
skips. This includes the independent TAA check and lighting regression. Raw log:
`ssgi-isolation/frozen-independent-taa/golden-final.log` (122.82 seconds).
The shared library also passes 484 tests with one existing ignored test after
updating the two shader-source assertions for repeated phase validation.

## Isolated GPU timing

Five alternating before/after process pairs per size, each with 24 warm-up and
120 profiled frames. No other local GPU test ran concurrently. These figures
cover the sum of five probe passes, not complete frame time. Profiling uses
blocking GPU readback, so wall-clock throughput is not a performance claim.

| Resolution | Before median (range), ms | After median (range), ms | Median delta |
| --- | --- | --- | --- |
| 256 x 256 | 0.097502 (0.095466-0.134262) | 0.108033 (0.105692-0.115772) | +0.010531 ms |
| 1280 x 720 | 0.388353 (0.382489-0.396046) | 0.432540 (0.428538-0.452480) | +0.044187 ms |

Most of the change is in phase-owner validation in the temporal pass: medians
0.039791 to 0.049613 ms at 256 square, and 0.195805 to 0.239524 ms at HD. This is
about an 11% increase in probe-pass cost on this fixture. The initial cold
256-square before run has visibly larger timing noise; all observations are
retained. These measurements do not establish a different adapter's budget.

## Open limits

- The HD TAA-jitter check independently fails with both shaders: mean RGB is
  1.48060330 before and 1.48050926 after, above its existing 0.75 bound. SSIM is
  0.97931148 before and 0.97931067 after. Both shaders pass this check at 256
  square. The HD failure predates this fix and remains open; the static HD pass
  is not a claim that HD temporal qualification is complete.
- The combined golden batch emitted invalid GPU timestamp totals. Its timing is
  excluded from this comparison; profiler reliability still needs investigation.
- The two portable-baseline Windows discrepancies, representative Bistro motion,
  other backends, and named RTX 4080 acceptance remain open.

Raw logs, commands, frozen executables, shader sources, phase captures, and
machine-readable timing observations are under
`tools/quality/out/windows-engine-plan/ssgi-isolation/`. Existing approved
baselines and thresholds remain unchanged.
