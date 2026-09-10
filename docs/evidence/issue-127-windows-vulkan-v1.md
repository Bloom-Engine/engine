# Issue #127 Windows/Vulkan path-tracing qualification

The AMD Radeon 760M passes the canonical PT hardware oracle and all four focused
temporal tests on Windows/Vulkan at clean source
`e8e08b90fc7cbad3e060d5ddb5c3ddb87aeb7add`. Native adapter metadata identifies
`AMD Radeon 760M Graphics`, `IntegratedGpu`, AMD driver `24.12.1 (LLPC)`, with
hardware ray query enabled. No approved golden, tolerance, or shader was changed.

## Canonical oracle

```powershell
$env:BLOOM_REQUIRE_RAY_QUERY = '1'
$env:BLOOM_GOLDEN_DIAGNOSTICS = '1'
cargo test --release --target-dir native/windows/target `
  --manifest-path native/shared/Cargo.toml --test golden_render `
  qualify_pt_oracle_hardware -- --ignored --exact --nocapture
```

The test passed with a real Vulkan ray-query device. It retains one device for
three progressive runs, three realtime camera-motion runs, and both fault controls.

| Mode | Repeats | Mean RGBA | Outlier pixels | Max error | SSIM | Render milliseconds |
| --- | --- | --- | --- | --- | --- | --- |
| Progressive, 300 frames | 3 | 0.134697 | 0.038147% | 69 | 0.997543544 | 1363, 239, 239 |
| Realtime camera motion, 48 frames | 3 | 0.060707 | 0.012207% | 48 | 0.999237806 | 70, 70, 71 |

Every image metric is identical across the three repeats. These short test
durations include different warm-up conditions and are not performance budgets.

Both seeded faults were rejected: BRDF energy produced mean error 5.307064 above
4.0; reprojection produced mean error 6.414501 above 6.0 and 6.071472% outlier
pixels above 1.0%. Their expected panic diagnostics are caught by the oracle;
the overall test exits zero only after normal runs pass and both faults fail.

The archive preserves normal accumulated/denoised output and intermediates
(radiance, albedo, normal, depth, visibility, motion, history, variance), plus
separate negative-control expected/actual/diff/heatmap images and JSON. Visual
inspection of both normal outputs found coherent geometry and illumination,
without black workgroup blocks or persistent block trails. Black background
outside scene geometry is intentional in this fixture.

## Focused temporal corpus

```powershell
$env:BLOOM_KEEP_TEMPORAL_DIAGNOSTICS = '1'
cargo test --release --target-dir native/windows/target `
  --manifest-path native/shared/Cargo.toml --test golden_render `
  realtime_path_tracing -- --nocapture --test-threads=1
```

All four tests passed. SVGF reported 10,984 accepted-history texels, 10,989 valid
and accumulated reprojections, and zero non-finite HDR pixels. All 855 moving
texels were classified: 45 retained history and 810 rejected it. Rigid motion
had zero severe-trail frames and zero frame-four coherent outliers. Camera reset
and PT off/on reproduced fresh seeds byte-for-byte. Lighting changes in both
directions had zero frame-12 coherent outliers.

This supplies the Vulkan hardware evidence missing from the existing
[Metal report](issue-127-pt-motion-requalification-v1.md). It does not establish
a pass for the full nine-scene quality corpus or the RTX 4080 profile. A later
broader golden batch found a Vulkan HAL counter defect and raster adapter skips;
that batch is explicitly not accepted as a broad-corpus pass.

Machine-readable metrics: [issue-127-windows-vulkan-v1.json](issue-127-windows-vulkan-v1.json).
Raw commands, logs, and captures are preserved in
`tools/quality/out/windows-engine-plan/issue-127-windows-vulkan-v1.zip` with a
SHA-256 sidecar. Publication and the complete issue acceptance audit remain
separate from these passing local hardware checks.
