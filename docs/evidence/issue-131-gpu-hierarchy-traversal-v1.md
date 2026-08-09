# Issue #131 GPU hierarchy traversal v1 evidence

This checkpoint qualifies Bloom's first GPU virtual-geometry hierarchy selector
at revision `b863d12106e755888ed22603416be9f7eaf8c9cc`. It selects bounded,
resident cluster records but does not yet decode/rasterize their cooked payloads
or replace ordinary model submission.

## Fixed ownership and ABI

The page pool now owns one additional fixed cluster table. Registration converts
the fully validated archive into GPU-addressable records before allocating or
publishing a mesh generation. A live mesh record is published only after its
cluster metadata and pinned root pages have been queued. Retirement quarantines
the cluster range alongside its mesh ID, page range, and physical slots until
GPU completion.

| Record | Exact bytes |
|---|---:|
| virtual mesh | 48 |
| logical page | 16 |
| cluster metadata | 128 |
| instance | 128 |
| selected cluster | 32 |
| missing-page request | 16 |
| traversal counters | 48 |
| traversal uniform block | 224 |

The cluster record retains object AABB/sphere, accumulated geometric error,
normal cone, primitive/material identity, page-local payload addresses, LOD and
triangle counts, and reciprocal parent/child group ranges. Registration rejects
assets exceeding the configured group width or the shader's fixed 32-level
traversal bound. Cluster-table exhaustion is preflight-atomic.

The selector itself owns fixed capacities for instances, selected clusters, and
missing-page requests. Construction validates all buffer, binding, storage
buffer count, workgroup, and dispatch limits before allocation. A selector is
bound to exactly one page pool and rejects another pool, a stale mesh, or a
retiring mesh before recording a dispatch.

## Conservative selection algorithm

One two-dimensional compute dispatch covers the maximum root prefix and every
instance. Only one root invocation proceeds for a multi-cluster atomic root
group. Each surviving invocation walks a bounded coarse-to-fine group chain:

1. transform every member's sphere with a conservative spectral-scale upper
   bound;
2. reject a group only when every member is outside an inward frustum plane;
3. compute maximum visible projected error using the nearest possible clip `w`;
4. refine only when the error exceeds the pixel threshold and every page of the
   complete child group is resident;
5. otherwise emit unique bounded missing-page requests and select the resident
   ancestor;
6. apply per-cluster frustum and conservative normal-cone rejection before
   writing selected records.

The scale bound uses a Gershgorin upper bound over `A^T A`, which is exact for
uniform orthogonal transforms and remains conservative for shear/non-uniform
scale. Normal-cone rejection is enabled only for finite, invertible, affine,
uniform-orthogonal instance transforms. Other valid transforms keep frustum and
LOD selection but skip cone rejection. Sphere angular extent is included in the
cone test, and cameras inside the sphere disable it.

Traversal is stateless. Camera cuts and fast instance motion cannot consume
stale occlusion or hierarchy state at this stage. Reaching the depth bound
selects the current resident group rather than dropping it.

## Metal GPU/CPU parity

Release tests ran through wgpu/Metal on the Apple M1 Max 32-core GPU. An
independent CPU implementation reads the validated archive rather than the
encoded GPU cluster mirror, then compares sorted selected records, page
requests, and every counter against GPU readback.

The three-level fixture proves:

- threshold 50 px selects four LOD-0 leaves after four group refinements;
- threshold 150 px selects two LOD-1 middle clusters;
- threshold 250 px selects two LOD-2 roots;
- an excluding frustum selects nothing and rejects both root groups;
- when only root and middle pages are resident, the two middle clusters remain
  selected while pages 3 and 4 are requested; two fallback groups are reported,
  with zero missing-current-page or invalid-record events;
- a two-record selected capacity reports four attempts and two overflows without
  writing past the fixed buffer;
- a one-record request capacity reports two attempts and one overflow;
- a unit normal cone selects two front-facing roots and rejects two back-facing
  roots; the same back view with non-uniform scale safely selects both because
  cone rejection is disabled;
- two instances select all eight expected leaves both before and after a large
  camera cut plus opposing 15/25-unit instance motion, with no requests,
  overflows, missing current pages, or invalid records.

The pool readback now additionally proves the complete cluster table is
byte-identical to the CPU mirror. In the deliberately small pool fixture:

| Fixed pool resource | Exact bytes |
|---|---:|
| physical pages: 3 x 4 KiB | 12,288 |
| page table: 16 x 16 | 256 |
| mesh table: 2 x 48 | 96 |
| cluster table: 32 x 128 | 4,096 |
| total | 16,736 |

## Regression boundary and gates

`GpuVirtualHierarchySelector` has no `Renderer`, `EngineState`, `ModelManager`,
frame-graph, scene, shader-library, or FFI owner. The ordinary renderer therefore
constructs no selector, cluster table, instance/output/request buffer, binding,
pipeline, pass, draw, or branch. Existing `GpuDrivenRenderer` geometry and
visibility composition are unchanged.

The governed quick lane passed in 52 seconds:

- 395 shared unit tests passed, one existing hot-reload test ignored;
- device negotiation passed;
- 59 golden tests passed, two hardware-specific tests ignored;
- four render-target tests, two visibility parity tests, and both visibility
  runtime tests passed;
- strict correctness/suspicious/performance Clippy, formatting, FFI/schema
  parity, CI contracts, and file-size governance passed;
- 39 quality-governance tests, three visual fault-engine tests, and 29 cooker
  tests passed;
- baseline wasm, native `models3d`, and wasm `web,models3d` checks passed;
- the no-default dependency graph still contains neither
  `bloom-geometry-format` nor SHA-256.

## Remaining acceptance work

No #131 acceptance checkbox changes at this checkpoint. The next slice must turn
selected resident clusters into bounded indirect visibility work and decode the
raw cooked page payload. Overflow must route the whole affected virtual batch to
the compatibility path rather than produce partial geometry. Conservative
previous-frame Hi-Z, asynchronous request readback/streaming, crack and motion
image corpora, 10-million-triangle stress, total GPU timings, fixed staging
overhead, and integrated/discrete adapter qualification remain open.
