# Issue #153: Radeon 760M Windows/Vulkan measurement

Two complete strict nine-case runs on the available Radeon 760M produced **7 passes and 2 visual failures** each. Both exited **1**; `repro-check` exited **0 (PASS)**. All 257 compared final/intermediate artifacts were byte-identical between runs, stable metadata matched, and timings stayed within the existing reproducibility bounds. Both host preflights and all 18 per-case postflights passed.

This is useful local qualification evidence for the available integrated GPU. It is not an RTX 4080 qualification or approval of Radeon performance budgets. No baseline images, visual thresholds, existing hardware budgets, or noise bounds were changed; neither run used `--report-only`.

## Source and host

- Measured commit: `8309985bf9f5358c38e5dd2cbc0e328d2bccb6f8`; both runs recorded a clean worktree.
- Issue's starting commit: `4508eaff082b849203ac60f9d8f3a327b71d95c9`. Windows execution/capture fixes were necessary before measurement.
- Branch: `codex/issue-153-radeon760m`.
- Manifest SHA-256: `95ef665ecd2734d527bd55738f69c9ab16bd472ad2239ab861c1bcc8a61ef320`.
- Windows 11 Pro x64, version 10.0.26200; Ryzen 5 7640HS, 6 cores / 12 logical CPUs; 27,704,946,688 bytes RAM reported.
- AMD Radeon 760M Graphics; Vulkan 1.3.292; AMD driver 24.12.1 (LLPC), Windows driver 32.0.12033.1030.
- Native telemetry confirms `hw-ray-query` GI in Sponza and Bistro. Other cases record their own selected GI path in telemetry, including `hiz-screen` and `hw-ray-query-pending`.
- Perry 0.5.1182, Cargo 1.96.1, Python 3.12.10, Node 24.21.0. The official Perry 0.5.1182 Windows archive was SHA-256 verified: `961d0789a50043b3e441c8826cfb8cc0106b3c4124d8fb9ca2381880961405d9`.
- The system Perry launcher points to a missing executable. The isolated toolchain lives under `tools/quality/out/toolchain/v0.5.1182/perry`. Perry 0.5.1220 was tried first, but its prebuilt standard library failed to link with undefined HTTP extension symbols; the failure log is included.

## Measurements

CPU/GPU frame p95 ranges below span the two final runs, in milliseconds. These are fixed scene workloads at the listed resolutions, not predictions of full-game FPS or 1080p performance. GPU work and CPU work overlap and should not be added together. Performance is observational until Radeon-specific budgets are reviewed.

| Case | Resolution / render scale | Result | CPU p95 ms | GPU p95 ms | SSIM |
| --- | --- | --- | --- | --- | --- |
| pbr-spheres-high | 512x512 / 1.0 | pass | 1.33-1.43 | 1.83-2.39 | 0.998787 |
| pbr-spheres-constrained | 512x512 / 0.75 | pass | 0.55-0.89 | 0.65-0.94 | 0.999385 |
| damaged-helmet | 512x512 / 1.0 | pass | 1.26-1.32 | 2.02-2.20 | 0.998639 |
| sponza-interior | 800x450 / 1.0 | fail | 1.71-1.75 | 3.46-3.50 | 0.972333 |
| bistro-exterior | 800x450 / 1.0 | pass | 2.33-2.40 | 7.80-7.82 | 0.987192 |
| skinned-alpha-motion | 800x450 / 1.0 | fail | 1.82-1.84 | 1.62-1.63 | 0.931998 |
| draw-light-stress | 1280x720 / 1.0 | pass | 9.43-9.89 | 11.64-11.73 | 0.999033 |
| weighted-transparency | 960x540 / 1.0 | pass | 1.99-2.69 | 5.94-5.95 | 0.999737 |
| masked-alpha-coverage | 960x540 / 1.0 | pass | 1.93-1.97 | 4.10-4.11 | 0.983675 |

## Remaining failures

- **Sponza:** SSIM `0.972333014`, required `>= 0.975000024`. Other visual metrics pass. Differences are concentrated around vegetation and lighting detail; the responsible renderer path has not yet been isolated.
- **Skinned/alpha motion:** SSIM `0.931997895`, required `>= 0.970000029`; luminance RMSE `0.047519531`, allowed `<= 0.039999999`. The largest visible differences are in the alpha-tested foliage surrounding the animated model. Repeat captures are identical, so this is reproducible rather than run-to-run timing noise.

The next useful renderer investigation is an owner-isolation comparison of alpha-tested foliage and its GI/shadow contribution in these two scenes, retaining the same assets, cameras, and thresholds. This evidence does not justify replacing the approved images.

## Fixes made while measuring

1. Resolve scene executables against their working directory on Windows and retain launch-failure evidence instead of aborting the runner.
2. Supply the owning HINSTANCE for Vulkan surfaces, including the native attachment path.
3. Export PNG colors according to the texture's actual RGBA/BGRA format. The first sphere capture had red and blue swapped; after the fix it passes the existing reference.
4. Honor headless capture on Windows using an offscreen renderer and fixed pixel dimensions. Before this fix, DPI/window borders changed 800x450 cases to 1178x619 and the 1280x720 stress case to 1898x1024. Those earlier timings describe different workloads and are diagnostic only.
5. Add a Radeon profile with adapter validation, native Windows CPU preflight/postflight counters, and explicitly unqualified performance measurements.
6. Keep the two hash-governed textual glTF fixtures at LF on Windows.

## Validation and evidence

- Asset/baseline check: PASS for all nine cases.
- Python qualification tests: 23 passed.
- Rust PNG round-trip test: passed for RGBA/BGRA, linear/sRGB formats and padded rows.
- Release native library and all eight unique scene executables built successfully.
- Two complete final runs, unchanged image checks: 7 pass / 2 fail each; no capture or telemetry errors.
- Reproducibility: PASS, 257/257 artifact comparisons byte-identical.
- Repository file-line gate still fails in nine unchanged files at the issue's pinned base; none of the edited files is a new violation. The exact output is included.

[First final run](../../tools/quality/out/windows-radeon760m-vulkan-corrected/summary.html), [repeat](../../tools/quality/out/windows-radeon760m-vulkan-repeat/summary.html), [reproducibility result](../../tools/quality/out/windows-radeon760m-repro/result.json).

The ZIP preserves repository-relative paths, approved references, all complete run directories, early failed-capture diagnostics, console/build/test logs, host inventory, the manifest, and a source patch. Generated toolchain binaries and third-party scene assets are excluded; their versions/revisions and hashes are recorded for fetching them again.

CPU snapshots passed before and after capture. Unrelated process starts and background GPU activity were not continuously monitored; no claim is made that absolutely no unrelated process started. The Radeon shares system memory, so dedicated-VRAM budgets are not inferred.

## Repeat locally

From the isolated worktree in PowerShell:

```powershell
$env:PATH = (Resolve-Path tools/quality/out/toolchain/v0.5.1182/perry).Path + ';' + $env:PATH
$env:PYTHONUTF8 = '1'
python tools/quality/run.py check
python tools/quality/run.py run full --machine-class amd-radeon760m-windows-vulkan --host-idle-timeout 600 --out tools/quality/out/radeon-next
```

Preserve exit code 1 and the full bundle when the known visual differences remain. The existing full RTX 4080 gate is still pending suitable hardware.
