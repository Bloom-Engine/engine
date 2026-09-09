# Issue #134 layered-PBR qualification

This checkpoint qualifies Bloom's shared layered material model against a
generated angular oracle, the realtime forward renderer, the visibility-buffer
composition path, the hardware ray-query path tracer, and a motion sequence.
The cross-path qualification is complete at revision `2b99693` on an Apple M1
Max / Metal adapter.

## Angular reference corpus

`bloom-brdf-reference --angular` generates the checked
`tools/bloom-reference/reference/layered-pbr-angular-v1.json` corpus. Its 63
rows cover base, specular/IOR, clearcoat, sheen, anisotropy, iridescence, and
combined materials at three view directions by three light directions per
scenario. Every row records the material, directions, individual BRDF lobes,
total `BRDF * NdotL`, PDF, and reciprocity error.

Regeneration is byte-exact. CPU component and reciprocity tolerances are both
`3e-5`; the measured maximum reciprocity error is zero. The corpus records five
intentional comparison boundaries: it excludes environment lighting and
post-processing, realtime finite-highlight compression is assessed in image
space, normal minification belongs to the motion corpus, stochastic path
transport uses converged-radiance tolerances, and iridescence uses Bloom's
bounded Khronos approximation rather than a spectral conductor model.

The checked corpus SHA-256 is
`110142a66bc22acaf0e7f6664a86f135b194fc62d203f1a92b685fbf2487f309`.

## Renderer parity

`layered_pbr_parity` selects the same `v1-l2` record for every lobe, disables
IBL, SSR, SSGI, SSAO, bloom, automatic exposure, motion blur, shadows, and
environment energy, then compares the 32x32 center region of the forward MRT
renderer and a 64-frame ray-query accumulation. It also compares each
path-traced lobe response with the linear oracle direction.

| Lobe | Forward/PT mean RGB MAE | Final RGB cosine | Luminance relative error | PT/oracle response cosine |
|---|---:|---:|---:|---:|
| Specular/IOR | 3.7106 | 0.999774 | 0.0779 | 0.969229 |
| Clearcoat | 6.5781 | 0.999883 | 0.1340 | 0.985841 |
| Sheen | 5.5430 | 0.997531 | 0.0743 | 0.981640 |
| Anisotropy | 20.4333 | 0.985835 | 0.2365 | 0.904537 |
| Iridescence | 3.8477 | 0.999813 | 0.0854 | 0.994356 |
| Combined | 8.3441 | 0.991555 | 0.1411 | 0.948319 |

The gates require mean display RGB MAE at most 24, final RGB cosine at least
0.96, relative display luminance error at most 0.30, and PT/oracle response
cosine at least 0.85 for significant linear responses. Smaller responses are
bounded to a maximum display response of 12. The worst measured values are the
anisotropy row, and all retain useful margin without weakening a threshold.

## Clearcoat-normal motion stability

`layered_pbr_motion` renders a high-frequency alternating tangent-space
clearcoat normal through Bloom's vector/variance mip chain while the camera
moves slowly, then subtracts the flat-normal control. The measured camera
motion mean RGB is `2.432037`, the filtered response mean is `0.205539`, the
maximum texture-specific adjacent-frame residual mean is `0.532235`, and the
coherent outlier-channel fraction is `0.0%`.

The hard sparkle gates are a maximum residual mean of `1.5` and maximum
coherent outlier fraction of `2%`. This test covers minification and temporal
stability independently from the static clearcoat-normal qualification in
`issue-134-clearcoat-normal-pt.md`.

## Visibility composition parity

Bloom's visibility renderer keeps unsupported/layered materials on the
authoritative forward compatibility path while visibility shading owns the
ordinary material population. The process-isolated parity fixture now places
moving specular/IOR, clearcoat, sheen, anisotropy, iridescence, and combined
layered draws over visibility-owned geometry. It verifies that the layered
specialization and sheen LUT initialize, that visibility mode records at least
the six layered compatibility draws, and that final depth ownership preserves
their results.

The forward and visibility runs compare the final display output and every raw
scene MRT attachment:

| Target | Changed components | Maximum delta | Mean delta |
|---|---:|---:|---:|
| Final RGBA8 | 150 / 81,920 | 1 code value | 0.00183105 |
| HDR scene RGBA16F | 1,838 / 81,920 | 0.000854492 | 0.000003760681 |
| Material RG8 | 19 / 40,960 | 1 code value | 0.000463867 |
| Motion RG16F | 55 / 40,960 | 0.000003815 | 0.000000005122 |
| Albedo RGBA8 | 80 / 81,920 | 1 code value | 0.000976562 |

The sub-LSB/f16 movement is the established manual perspective-reconstruction
rounding allowance; no target exceeds its existing parity gate. Together with
the lobe-response forward/PT/oracle test above, this closes the remaining
forward/visibility/path-traced agreement criterion without adding a shipping
pass, allocation, shader branch, or binding.

## Regression and cost qualification

`./scripts/ci-check.sh --quick --summary
target/ci/quick-issue-134-final.json` passed at the qualified revision in 25
seconds:

- 328 shared unit tests passed, 1 intentionally ignored;
- 59 runnable GPU goldens passed, 2 intentionally ignored;
- all 4 render-target integration tests passed;
- strict lint, formatting, FFI/schema parity, web/wasm compilation, quality
  governance, visual-fault controls, and 20 canonical examples passed;
- 19 release tests and strict Clippy passed for `bloom-reference`.

The angular generator and both renderer gates add no shipping render pass,
image, shader branch, GPU allocation, binding, or per-frame work. Their
production runtime and binary shader cost is zero.

The 2026-08-10 targeted Metal gates passed: both process-isolated visibility
tests, the visibility runtime test, and the 64-frame hardware ray-query
layered parity test. The repository quick wrapper stopped before its test phase
because the current branch's pre-existing `renderer/mod.rs` is 20 lines above
its stored file-size baseline; this qualification does not modify that file.

## Commits

- `eca754e` — clearcoat-normal motion stability gate;
- `ad97e77` — versioned layered-PBR angular reference corpus;
- `21356b8` — forward, path-traced, and oracle parity gate;
- `2b99693` — complete layered forward/visibility MRT parity gate.

Machine-readable measurements accompany this note in
`docs/evidence/issue-134-layered-pbr-qualification.json`.
