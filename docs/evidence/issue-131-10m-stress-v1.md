# Issue #131 10M source-triangle stress v1 evidence

This checkpoint qualifies Bloom's opt-in virtual-geometry runtime on a
deterministic 10,000,000-source-triangle static workload at revision
`741cc5ae874ba617988227a73cccf7ab24aa3ba0`. The gate exercises hierarchy
selection, bounded indirect emission, streaming feedback, fixed physical
residency, visibility-buffer shading, camera motion, and GPU timestamps in the
production renderer integration.

## Deterministic workload

`tools/quality/prepare_virtual_geometry_stress.py` creates 100 independently
placed glTF meshes. Each mesh owns one 100,000-triangle primitive while all
primitives share a compact deterministic grid buffer. The source closure is
therefore only 1,819,732 bytes while still presenting 10,000,000 authored
indexed triangles and 100 independently filtered hierarchy instances to the
cooker and runtime.

The source hashes are:

- glTF: `eba12b4d682f587a177f4a556aa3469c03d03ea01c8d35374a4a9d5b0f1c5c57`;
- binary: `823929f38f2c63625b3156019d0de25c076bd0fb6672b43b649a6934502abe49`.

The version-2 `quantized32` artifact is 582,052,704 bytes. It contains 245,500
clusters in 8,496 pages and reaches LOD level 7. Its 1,500 coarse roots occupy
50 pages and require 3,276,800 resident bytes. The complete artifact SHA-256 is
`45b6aa47ce817589911d33f3f4b32387847cf28a9baf6c94523ae1f81db198d8`,
the payload SHA-256 is
`1a9ca79489830d81211f54f2a9b2e55446a77b9ee5f36fe9ad1490c699001e13`,
and the source-closure SHA-256 is
`d28cc38f1fa6568e38f8a5ed29ecacdda2507af0995d4b0aa463e76f38038649`.

## Traversal cost corrections

Two costs exposed by this workload were corrected before qualification:

1. A source-filtered instance dispatched against every root in its shared
   archive and rejected unrelated roots in the shader. The cooker now groups
   roots by source mesh. Runtime assets cache a backward-compatible covering
   span per source, and each instance dispatches only that span while retaining
   the shader identity check for older interleaved archives.
2. Previous-frame Hi-Z projected eight bounding-box corners twice for every
   cluster in an atomic group. Traversal now unions the intersecting clusters'
   local AABBs and performs one conservative previous/current projection for
   the complete group. A group is rejected only when its union is proven
   occluded; uncertainty still fails open. The existing real-GPU camera-motion
   Hi-Z oracle retains its exact visible/occluded expectations.

On the same Apple M1 Max Metal path, hierarchy-selection GPU mean fell from
4.5010 ms in the diagnostic baseline to 1.7576 ms. Total GPU frame mean fell
from 6.6660 ms to 4.7957 ms.

## Fixed-budget motion and timing gate

The environment-gated release test warms streaming for 180 moving-camera
frames, then profiles 120 further moving-camera frames at 640x360 with
visibility-buffer shading. The physical pool is a hard 64 MiB. It settled at
955 resident pages, or 62,586,880 physical bytes, after uploading 905 streamed
pages. The final frame contained 10,220 selected clusters and no visible tile
holes.

| Measurement | Result | Enforced maximum |
|---|---:|---:|
| Wall frame mean | 8.2794 ms | 16.6667 ms |
| GPU frame mean | 4.7957 ms | 8.0 ms |
| GPU frame p95 | 7.9175 ms | 12.0 ms |
| Hierarchy selection GPU mean | 1.7576 ms | 3.0 ms |
| Draw emission GPU mean | 0.1224 ms | 0.5 ms |

CPU frame mean was 3.1560 ms and CPU p95 was 8.7285 ms. The runtime reported
zero fallback groups, missing-current pages, selected overflow, request
overflow, invalid records, depth-limit fallbacks, and pending streaming groups.

## Automated qualification

- full shared library: 459 passed, one existing ignored;
- complete real-GPU golden renderer corpus: 77 passed, two expected ignored;
- deterministic stress generator tests: pass;
- 10M stress gate with 180 warmup and 120 measured frames: pass;
- cooker: 32 passed, strict `-D warnings` Clippy passed;
- quality governance/fault corpus: 41 passed;
- shared strict correctness/suspicious/performance Clippy: pass;
- WebAssembly `web` check and native no-default-feature check: pass;
- formatting, diff whitespace, and example inventory: pass.

The repository file-size ratchet remains red only for the same three unrelated
pre-existing files: `renderer/shaders/post.rs`, `renderer/shaders/ssgi.rs`, and
`tests/golden_render/temporal_history.rs`. This checkpoint adds no overage.

## Remaining #131 work

This closes the 10M-source-triangle fixed-residency and integrated-Metal timing
slice. Runtime activation remains opt-in. The next independent milestone is
asynchronous #136 store/index-backed archive I/O behind page feedback, followed
by discrete-GPU and Metal/Vulkan/Direct3D 12 timing and quality evidence.
