# Issue #131 virtual draw emission and raw-page decode v1 evidence

This checkpoint qualifies Bloom's first bounded virtual-cluster indirect
stream and cooked-page GPU decoder at revision
`92daf0eae684ca5a0b359cc8c991f728ffb83600`. It proves that selected resident
clusters can become executable indirect work without expanding their cooked
vertices. It does not yet register a production render path or change ordinary
scene pixels.

## Fixed ownership and ABI

`GpuVirtualDrawEmitter` is constructed explicitly from one hierarchy selector.
It rejects another selector, validates command-buffer and workgroup limits
before allocation, and owns three fixed resources:

| Record | Exact bytes | Purpose |
|---|---:|---|
| non-indexed draw command | 16 | triangle vertex count and selected-record address |
| emission state/count | 48 | indirect count, whole-batch fallback, and counters |
| compute dispatch command | 12 | bounded indirect emission dispatch |

The command capacity exactly equals the selector's selected-cluster capacity.
The prepare pass reads selection counters and publishes a bounded draw count and
dispatch. The emit pass writes one command per admitted cluster; its
`first_instance` is the matching selected-record index. Triangle counts come
only from the strictly validated archive and remain bounded by the 4 MiB page
ceiling.

The state count lives at byte zero for `multi_draw_indirect_count`. That optional
feature is not assumed for production activation: adapters without it must use
a separately qualified bounded submission strategy or remain on compatibility
rendering. This checkpoint's Metal raster oracle uses the supported fixed-count
form with the fixture's known four commands.

## Whole-batch safety

The prepare shader suppresses the complete virtual batch when selection
overflowed, the selector observed an invalid record, a current selected page is
missing, or selected/count capacities disagree. Invalid-plus-missing telemetry
saturates instead of wrapping. No partial geometry is published as a valid
batch.

The real Metal overflow fixture attempts four leaf selections against a
two-record capacity and proves:

- `selected_count = 4`, `selected_overflow = 2`;
- `draw_count = 0`, `batch_fallback = 1`;
- indirect dispatch x is zero;
- emitted draw and triangle counters remain zero.

Request overflow has different semantics. With only complete resident middle
groups available, two leaf-page requests are attempted against one request
slot, but the two resident ancestors remain a complete valid fallback batch.
The emitter therefore publishes two draws with no batch fallback. This prevents
feedback pressure from creating a visible hole.

## Raw cooked-page vertex pulling

The common WGSL decoder consumes the fixed physical page buffer directly. It
supports both validated archive encodings:

- version 1: 72-byte Float32 position, normal, tangent, UV0, UV1, and color;
- version 2: 32-byte quantized AABB-local position, octahedral normal/tangent,
  binary16 UV0/UV1, UNORM8 color, handedness, and the missing-tangent flag.

It reads each cluster's local `u8` corner index, page-local vertex/index offsets,
stride, AABB, and mesh encoding from the production GPU tables. The Metal probe
selects four resident leaves, decodes all twelve indexed corners for each
format, and compares every output lane plus selected/cluster/corner/local-index
identity. Float32 and quantized positions, normals, tangents, UV sets, and
colors match their CPU-authored values within `1e-5`.

## Executable Metal indirect proof

A real render pass consumes the GPU-written command buffer in the same command
encoder after hierarchy selection and emission. Four indirect triangles write
four distinct `Rgba8Uint` pixels. Readback is exactly `[1, 2, 3, 4]` in the red
channel, proving that every command executed and that `first_instance` retained
the expected selected-record index.

Together with the independent traversal oracle, the focused virtual-geometry
suite now has 22 passing tests covering strict load, fixed residency, retirement,
GPU table identity, LOD/frustum/cone selection, missing-page fallback, bounded
overflows, camera cuts and motion, both vertex formats, command emission, and
actual indirect rasterization.

## Regression boundary and gates

The pool, selector, emitter, and decoder have no `Renderer`, `EngineState`,
`ModelManager`, frame-graph, scene, FFI, or ordinary glTF owner. The shipping
path therefore constructs none of their buffers, pipelines, bindings, passes,
draws, or branches. Existing rendering pixels, allocations, and steady-state
work are unchanged.

The complete governed quick lane passed in 51 seconds on the qualified tree:

- 401 shared unit tests passed and one existing hot-reload test was ignored;
- the negotiated real-device test passed;
- 59 golden tests passed and two hardware-specific tests were ignored;
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

No #131 acceptance checkbox changes at this checkpoint. Selected virtual work
still needs current/previous transform and material-remap records, integration
with #27's exact PBR visibility composition, a bounded non-indirect-count path,
conservative previous-frame Hi-Z, asynchronous request readback/streaming,
crack and temporal image corpora, 10-million-triangle stress, per-MRT parity,
total GPU timings, and integrated/discrete adapter qualification.
