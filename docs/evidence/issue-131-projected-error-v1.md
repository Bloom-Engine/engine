# Issue #131 projected-error geometry evidence v1

This checkpoint qualifies Bloom's cooked virtual-geometry error metric at
revision `6ea4c5b3ce6430e15c9deeeea169236e46400ce3`. It corrects an
underestimate in the value used by the runtime's projected-pixel LOD selector
without changing the public rendering API or adding a runtime pass,
allocation, upload, or shader branch.

## Fault and correction

The meshoptimizer simplifier returns a useful quadric error metric, but that
value is not a strict one-sided positional bound. An independent oracle found
a real hierarchy edge whose maximum LOD0-vertex distance to the parent
triangle surface was `0.00053679076`, while the stored error was only
`0.00040672`. The runtime could therefore accept that parent at a distance
where its measured source-vertex deviation exceeded the selected pixel
threshold.

Geometry recipe v2 retains the simplifier result, builds an independent
triangle BVH for every accepted simplification, measures every source-group
vertex against the simplified triangle surface, and uses the larger value.
The edge error is then accumulated with the maximum child error as before.
The recipe version was incremented because corrected error records change
artifact bytes while the `.bgeo` wire format remains version 2.

The automated hierarchy oracle is intentionally independent of the production
BVH. For every parent group in a multi-level 32x32 test surface it:

- walks every LOD0 descendant;
- measures each descendant vertex against every triangle in the parent group;
- requires the stored object-space error to upper-bound that measurement; and
- projects both values at the exact one-pixel selection distance and requires
  the measured deviation to remain at or below one pixel.

The original cooker fails this oracle at the edge above. Recipe v2 passes it.
Existing real-GPU traversal tests separately prove that the Metal selector and
its CPU oracle make the same projected-error LOD decisions.

## Determinism and offline cost

Two independent clean recipe-v2 cooks of the deterministic 10M-source-triangle
asset produced byte-identical artifacts, manifests, and indexes:

| Record | SHA-256 |
|---|---|
| Source closure | `d28cc38f1fa6568e38f8a5ed29ecacdda2507af0995d4b0aa463e76f38038649` |
| Geometry artifact | `13ab0d150328b9f67ebc8abbe0d73891a10d72033bc3a18b4d9f6a8364cbd864` |
| Geometry payload | `1a9ca79489830d81211f54f2a9b2e55446a77b9ee5f36fe9ad1490c699001e13` |
| Profiled manifest | `41559b4fea12bb32fa111c6010af4540114770ae7ec98520c743335f67c43032` |
| Store index | `5c69a9a6cbf979d4c8483b3d8adc980cbe23d73862d5a79457051fe626d053c8` |

The recipe-v2 build key is
`f13f69fe60a03ca7014d49ede75b83d26ef77e036deb8f253f5150421c2f74ef`.
Both clean cooks retained 245,500 clusters, 8,496 pages, and a 582,052,704-byte
artifact. Their wall times were 92.35 and 100.27 seconds. The independent BVH
therefore did not regress the prior 121.41-second reference cook on this host.

## File-backed 10M runtime regression gate

The corrected artifact passed the existing file-backed Metal gate on an Apple
M1 Max: 180 moving-camera warmup frames followed by 120 measured frames at
640x360, with the physical pool fixed at 64 MiB.

| Measurement | Recipe v2 | Enforced maximum |
|---|---:|---:|
| Wall frame mean | 6.0585 ms | 16.6667 ms |
| GPU frame mean | 4.3993 ms | 8.0 ms |
| GPU frame p95 | 7.7775 ms | 12.0 ms |
| Hierarchy selection GPU mean | 1.6000 ms | 3.0 ms |
| Draw emission GPU mean | 0.1091 ms | 0.5 ms |

The run settled at 955 resident pages, uploaded 905 streamed pages, completed
370 file requests with zero I/O failures, and read 75,965,888 bytes. It ended
with zero fallback groups, missing-current pages, selected/request overflow,
invalid records, depth-limit fallbacks, and pending streaming groups. The
selected cluster count remained 10,220, so the corrected metadata caused no
runtime quality or performance regression on this deterministic workload.

## Automated qualification

- cooker release tests: 34 passed;
- cooker strict `-D warnings` Clippy: passed;
- independent projected-error hierarchy oracle: passed;
- two clean 10M cooks: byte-identical artifact, manifest, and index;
- strict asset-index inspection: passed;
- file-backed 10M Metal stress gate: passed;
- formatting and diff whitespace: passed.

The file-size ratchet remains red only for the same three unrelated
pre-existing files: `renderer/shaders/post.rs`, `renderer/shaders/ssgi.rs`, and
`tests/golden_render/temporal_history.rs`. This checkpoint adds no overage.

Discrete Vulkan and Direct3D 12 qualification is still pending because the
repository currently reports no online self-hosted runners. The queued
cross-backend workflow is
<https://github.com/Bloom-Engine/engine/actions/runs/33046639665>.
