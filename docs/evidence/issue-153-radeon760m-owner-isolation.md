# Issue #153: Windows baseline portability investigation

The source revision that produced the approved baselines (`09ad0b755af9f10083712327d7f0edb1d88f228b`) and the current renderer produce **byte-identical final PNGs** for Sponza and skinned/alpha motion on this Windows Radeon 760M host. Both retain the same strict failures against the approved portable images. Under this tested configuration, these are not regressions introduced since that baseline source.

This comparison uses the old renderer, loaders, shaders and scene sources from its checkout, with the current Windows launch/headless corrections and qualification orchestrator overlaid to execute it on this host. The current corpus manifest and approved PNGs were copied in so the comparison uses identical inputs and thresholds. The original/current resolution, frame counts, seed, timestep, render scale, camera, settings and thresholds were independently compared and match for both cases. The exact adaptation patch is preserved at `tools/quality/out/radeon-evidence/windows-oracle-adaptations.patch`. Native build records identify `engine-quality-baseline/native/shared` and `engine-quality-baseline/native/windows`. The old-source checkout is a diagnostic oracle, not a clean qualification commit.

## Controls

| Configuration | Sponza SSIM | Skinned/alpha SSIM | Skinned/alpha luminance RMSE |
| --- | --- | --- | --- |
| windows-radeon760m-vulkan-corrected | 0.972333014 | 0.931997895 | 0.047519531 |
| radeon-owner-software-gi | 0.969276249 | 0.931530952 | 0.047676153 |
| radeon-owner-bound-materials | 0.972333014 | 0.931997895 | 0.047519531 |
| radeon-owner-dx12-software-gi | 0.970000505 | 0.931594729 | 0.047674358 |
| radeon-owner-anisotropy-1 | 0.952164710 | 0.923400104 | 0.048658907 |
| radeon-owner-baseline-source | 0.972333014 | 0.931997895 | 0.047519531 |

- Forcing software GI retains both errors, so simply changing GI is not a fix.
- Forcing bound material bindings produces exactly the current qualification images.
- DirectX 12 with software GI produces essentially the same failures as Vulkan with software GI. This is an image-only diagnostic; it does not claim a governed DirectX timing run.
- Reducing the material sampler from 16x to 1x anisotropy worsens both images. This temporary diagnostic was reverted. Its patch is preserved; no reduced-quality sampler setting is shipped.
- The old-source/current byte equality rules out the intervening Three.js material compatibility changes as the cause in these cases.

The remaining difference is between the approved portable baseline and this Windows/GPU/compiler configuration. This does not yet identify a specific driver, compiler or rasterization cause, and it does not authorize a baseline or threshold update. Further cross-platform normalization work needs a separate renderer investigation or a human-reviewed backend-specific baseline decision. The qualification task explicitly accepts a complete strict failure bundle.

## Detection and final state

The negative-control command detected all five governed seeded faults: BRDF energy, shadow placement, GI leakage, motion history and texture orientation. Its exit code was 0 and `result.json` records all five as detected.

The Windows fixes remain on `codex/issue-153-radeon760m`. No experimental rendering changes remain in that branch. The two full runs still provide the definitive nine-case Windows/Radeon qualification: seven passes, two strict visual failures, all required hardware ray-query cases measured, and reproducibility PASS with 257 byte-identical artifacts.
