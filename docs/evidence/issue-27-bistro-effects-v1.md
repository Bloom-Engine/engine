# Issue #27 full-Bistro effects parity v1

This checkpoint qualifies the opt-in visibility composition path against the
ordinary forward path at exact revision
`24d8ba4af7231a4e81c8829a6049b08da19f35e8`. The clean-revision test loads
the complete `bistrox.gltf` corpus, attaches all 2,909 placements, settles the
temporal/ray histories, and drives the same 30-step camera route in isolated
forward/off and visibility/shade processes.

## Large-scene blocker and fix

The first full-scene run exposed a production limit that the synthetic corpus
did not reach. Bistro uses 1,738,262 vertices in Bloom's 96-byte `Vertex3D`
storage ABI: 166,873,152 bytes (159.143 MiB). Apple M1 Max Metal limits one
storage-buffer binding to 134,217,728 bytes, so binding the entire 256 MiB
vertex arena made visibility-shader bind-group creation fail validation.

Visibility reconstruction now exposes the used vertex range through three
offset-aligned storage windows. Each full window is the largest common
multiple of the 96-byte vertex stride and the adapter's storage-offset
alignment that fits within `max_storage_buffer_binding_size`. WGSL routes the
global vertex index through `arrayLength`-derived segment boundaries, so the
existing draw/index/base-vertex namespace is unchanged. Three windows also
respect this adapter's exact limit of nine fragment-stage storage buffers.
The resource cache includes the used vertex-byte range, so an arena that grows
without changing allocation generation still rebuilds the bindings.

A unit regression uses Bistro's exact vertex count against a 128 MiB binding
cap. The existing real-GPU visibility/MRT oracle remains green after the
segmentation: its final image differs in 187 of 81,920 channels, all by one
code value, with mean delta 0.00228271.

## Full-Bistro protocol

- Apple M1 Max integrated GPU, Metal, negotiated high-end tier;
- 640x360 native render resolution and quality preset 4;
- all 2,909 glTF mesh placements attached;
- 108 stationary settle frames, followed by 30 camera-motion steps;
- captures at steps 0, 5, 10, 15, 20, 25, and 30;
- TAA, SSAO, SSR, SSGI, bloom, directional shadows, and the ordinary forward
  compatibility pass active;
- hardware ray-query SSGI, with valid TAA, SSR, and SSGI histories;
- separate cascaded-shadow-map and virtual-shadow-map process pairs;
- final-camera visibility telemetry: 2,404 eligible draws and 164
  compatibility draws.

The oracle disables asynchronous renderer occlusion. Its previous-frame GPU
readback can become CPU-visible one frame earlier or later when the A/B pass
cost differs, which changes draw admission before shading is compared. That
is an occlusion scheduling/qualification concern, not a visibility evaluator
delta; occlusion therefore remains an explicit separate gate.

## Results

Metrics compare RGB output after the complete screen-effect stack. Values are
mean and RMS code-value delta, 99th-percentile and maximum code-value delta,
and 8x8-window luminance SSIM.

| Shadow | Step | Mean | RMS | P99 | Max | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| CSM | 0 | 0.007268519 | 0.085255607 | 0 | 1 | 0.999949947 |
| CSM | 5 | 0.027641782 | 0.215159017 | 1 | 14 | 0.999828217 |
| CSM | 10 | 0.025400752 | 0.236122345 | 1 | 18 | 0.999738925 |
| CSM | 15 | 0.064822049 | 0.469083215 | 2 | 15 | 0.999430423 |
| CSM | 20 | 0.044548611 | 0.303008640 | 1 | 19 | 0.999572590 |
| CSM | 25 | 0.064280961 | 0.317713456 | 1 | 18 | 0.999355183 |
| CSM | 30 | 0.058449074 | 0.337812784 | 1 | 19 | 0.999529795 |
| VSM | 0 | 0.021362847 | 0.340015488 | 1 | 22 | 0.998759431 |
| VSM | 5 | 0.026597222 | 0.207714358 | 1 | 14 | 0.999824722 |
| VSM | 10 | 0.025357350 | 0.235287575 | 1 | 13 | 0.999728743 |
| VSM | 15 | 0.068846933 | 0.486732242 | 2 | 15 | 0.999385902 |
| VSM | 20 | 0.044612269 | 0.303104118 | 1 | 19 | 0.999572056 |
| VSM | 25 | 0.064267940 | 0.317692964 | 1 | 18 | 0.999355197 |
| VSM | 30 | 0.057352431 | 0.334412316 | 1 | 19 | 0.999534061 |

Both seven-frame pairs pass their permanent gates. The VSM route also requires
active virtual-shadow telemetry, non-zero residency and demand, and cache hits;
a settled cache is allowed to render zero new pages on the final frame.

## Scope and remaining gates

This is evidence for the full-Bistro camera/effects/shadow portion of #27 on
an integrated/tile-based adapter. It does not qualify custom opaque materials,
planar probes, the full skinning/refraction/transparency stress corpus,
asynchronous occlusion parity, or a discrete adapter. The measured performance
activation gates also still fail. Visibility shading therefore remains opt-in
and is not enabled by default.

Clean-revision checks passed:

- `visibility_buffer_bistro_runtime`: complete CSM and VSM matrix;
- `visibility_buffer_parity`: two real-GPU final/MRT tests;
- nine visibility-buffer unit/GPU tests;
- `cargo fmt --all -- --check`.

Strict Clippy is currently blocked by 188 repository-wide pre-existing
current-toolchain findings; this change does not claim that unrelated debt as
part of its qualification.
