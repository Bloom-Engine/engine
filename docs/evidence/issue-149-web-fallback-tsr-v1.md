# Issue #149 Web fallback TSR v1 evidence

This checkpoint extends the governed temporal-quality path from native Metal
to the production browser WebGPU fallback renderer. The exact comparison base
is `26b1893`.

## Defect exposed by the real 3D browser gate

The previous browser smoke rendered only a known 2D clear. It passed at the
comparison base even though the first real 3D frame could not construct the
SSR temporal pipeline. Loading the canonical Damaged Helmet and enabling the
normal quality-preset-3 TAA, SSGI, SSR, and shadow path exposed Chrome's
validation error:

```text
'dpdy' must only be called from uniform control flow
... control flow depends on possibly non-uniform value ... off_screen
... CreateRenderPipeline ... ssr_temporal_pipeline
```

`depth_gradient` was evaluated after the fragment-specific off-screen history
early return. The fix moves the existing `dpdx`/`dpdy` footprint evaluation
before that return, where derivative evaluation is uniform across the fragment
quad. The accepted path does not change on-screen SSR math, sampling kernels,
bindings, textures, buffers, render-graph topology, or persistent memory. An
Naga parse-and-validate unit test and source-order assertion keep the derivative
outside divergent control flow.

## Governed real-browser sequence

The v2 Chrome gate retains the known-color presentation oracle and adds three
matched eight-frame camera sequences after 12 warm-up frames:

- native 1.0 with TAA;
- fractional 0.75 with TAA;
- fractional 0.75 without TAA as an independent spatial control.

All runs import the canonical `DamagedHelmet.glb` through Bloom, exercise
textured and primitive geometry plus thin rails, and use the software SSGI and
SSR fallback. The final independent run reports:

| Metric | Result | Limit |
|---|---:|---:|
| Fractional-to-native RGB frame RMSE | 0.027498781 | <= 0.030000000 |
| Fractional-to-native motion-derivative RMSE | 0.024833870 | <= 0.027000000 |
| Native motion activity, RGB 8-bit | 0.225245 | >= 0.050000 |
| Fractional motion activity, RGB 8-bit | 0.625422 | >= 0.050000 |
| Fractional scene luma standard deviation | 68.807094 | >= 12.000000 |
| No-TAA fractional-to-native frame RMSE | 0.029252788 | diagnostic |

The earlier accepted run measured 0.027499950 frame RMSE and 0.024833868
motion-derivative RMSE. The independent result is effectively identical and
well inside both limits.

The browser capability snapshot proves that this is the intended fallback:

- backend `browserwebgpu`, selected capability tier `modern`;
- hardware ray query `false`, active SSGI backend `hiz-screen`;
- Tier-B deterministic paged material path;
- one imported material, one mesh, five textures, two samplers, and two buffer
  views, with zero stale or limit fallbacks.

CI now uploads `target/ci/web-smoke` on success and failure. The artifact owns
`result.json`, adapter/runtime capability evidence, browser logs, the known 2D
frame, and all three numbered 3D sequences.

## Performance and resource contract

Exact release binaries at base and candidate were alternated for three
quality-preset-3 PBR-spheres profiles. Every run used 120 warm-up and 300
measured fixed-timestep frames with the same camera and requested 512x512
surface at render scale 1.0.

| Revision | GPU frame means (ms) | Mean | GPU p95 mean | SSR-pass mean |
|---|---|---:|---:|---:|
| Base `26b1893` | 33.856949, 33.373425, 34.510888 | 33.913754 | 51.424834 | 3.043760 |
| Candidate | 33.652467, 34.318845, 32.828617 | 33.599976 | 51.739695 | 3.017834 |

Candidate mean GPU time is 0.93% lower, SSR-pass mean is 0.85% lower, and GPU
p95 is 0.61% higher. The ranges overlap and pairwise noise changes direction,
so the result is classified as no measurable regression, not an improvement.
All six telemetry reports show zero steady-state bind-group creation, graph
compiles, first-use pipeline creation, and transient physical resource
creation.

## Validation

- real Chrome browser gate: two independent final passes;
- quality-contract lane: 47 Python governance tests, three `bloom-diff` tests,
  and 48 cooker tests passed;
- complete shared library: 483 passed, one ignored;
- complete real-GPU golden renderer corpus: 86 passed, three ignored;
- WebAssembly `wasm32-unknown-unknown` web-feature check passed;
- headless device negotiation/capability test passed;
- Rust formatting and diff whitespace checks passed.

The only native warning is the pre-existing unused `mut` in `src/drs.rs`;
WASM reports existing target-specific unused-code warnings.

This advances the Web/fallback acceptance criterion for issues #149, #128,
and #135. It is not closure of the parent quality goal: hosted Vulkan hardware
evidence and the remaining representative-scene/platform matrix still remain.
