# SSGI depth receivers and resolution-independent normals

The Radeon 760M/Vulkan HD stationary-TAA fixture exposed two surface
reconstruction errors. Correcting them together passes the existing 16-frame
warm-up and image-change bounds and removes the horizontal bands in the
captured indirect-light buffer. Neither the warm-up nor its thresholds changed.
Full-corpus and cross-platform qualification is still in progress.

## Cause and correction

Probe placement requested depth at fractional screen coordinates and then used
those coordinates to reconstruct a position. Point sampling returns the depth
of a particular texel; that depth belongs to its center. On an inclined plane,
using a different coordinate moves the reconstructed receiver off the surface.
Placement now reconstructs from the selected texel center. Resolve and its
capture-only geometry diagnostics use the same coordinate convention.

The normal came from the cross product of two finite-difference surface edges.
Its magnitude shrinks with pixel area, but the normalization guard compared its
squared length against a fixed `1e-6`. Valid small footprints could therefore
receive a camera-facing fallback normal. The correction scales each edge by
its largest absolute component before taking the cross product. The existing
degeneracy guard then measures angular degeneracy independently of footprint
scale; other uses of the direction helper retain their existing behavior.

Both placement and resolve use this surface-normal helper. The diagnostic
resolve pass also uses it, so reported geometry matches production.

## Isolated controls

These HD captures use the same scene, original 16-frame warm-up, Vulkan adapter,
and software Hi-Z GI path. The measured buffer is raw half-resolution SSGI
before final TAA, at 640×360 for a 1280×720 render.

| Variant | Mean RGB change | Edge change | Outlier pixel fraction | SSIM | Existing gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Original | 1.480509259 | 0.003520461 | 0.001032986 | 0.979310670 | Fail |
| Texel centers only | 1.529759838 | 0.003519964 | 0.000746528 | 0.976575446 | Fail |
| Scaled surface normal only | 0.879440104 | 0.003198458 | 0.003294271 | 0.973574972 | Fail |
| Both corrections | 0.336552373 | 0.001203517 | 0.000881076 | 0.990864689 | Pass |

The original fails at 16 frames and passes a 32-frame diagnostic control.
Increasing warm-up would leave its incorrect surface geometry and visible bands intact.
The combined correction also passes the original 256×256 stationary-TAA
fixture and preserves a byte-identical settled, non-jittered angular cycle.

## Regression coverage and remaining verification

The HD fixture is now a regular GPU regression, rather than depending on the
optional `BLOOM_SSGI_PROFILE_HD` environment variable. A second regression runs
the actual production placement pipeline against an analytic inclined plane.
It checks receiver positions and normals at four placement phases for depth
extents 128×128, 640×360, 641×361, and 1920×1080. On Radeon/Vulkan all 154,720
receiver checks pass: maximum plane error is below `4.8e-7` world units and
maximum normal-component error is below `1.4e-4`.

Restoring the old coordinates independently fails the analytic test with a
`0.016759634` world-unit plane error. Restoring the old normal guard independently
fails with a `0.6` normal-component error. Both temporary controls were reverted.
Verified Radeon/DX12 runs also pass all 154,720 analytic checks and all three
Hi-Z fixtures. DX12's HD mean RGB change is `0.401780961`, edge change
`0.001269536`, outlier fraction `0.000703125`, and SSIM `0.985344129`.

The golden harness selects its adapter using `WGPU_BACKEND`. Earlier nominal
DX12 diagnostic runs set only `BLOOM_WGPU_BACKEND` and actually used Vulkan;
those runs do not qualify DX12. The subsequent verification records both
selectors and asserts the actual logged adapter is `Dx12`.

Diagnostic commands, source patches, per-variant logs, and captures are retained
under `tools/quality/out/windows-engine-plan/hd-depth-coordinates/` and
`hd-depth-normal-control/`. Validation of the production correction is retained
under `ssgi-surface-fix/`. The isolated experiments record parent source
`662a44f85b23fd91cff53b6479e6373f208dc321` plus their exact temporary patches;
their original source bytes were restored after each experiment.

The complete local shared suite passes, including all 488 library tests and
93 reported golden passes. Four goldens remain ignored, and optional external
fixtures do not establish acceptance when they return early. The existing
bright/dim/bright lighting-cycle regression retains 48 byte-identical settled
frames and recovers the independent fresh-reference image.

Broader scene-image coverage, other backends, and timing qualification remain
separate checks. This report does not claim that
the entire #135/#149 temporal reconstruction acceptance contract is complete.
