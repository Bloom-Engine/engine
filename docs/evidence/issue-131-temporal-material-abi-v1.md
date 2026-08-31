# Issue #131 temporal and material ABI v1 evidence

This checkpoint qualifies the render-state boundary required before virtual
clusters can enter Bloom's visibility/PBR path. The qualified code revision is
`1b4d82816c09f6a544fd8e3d4db77801e84f0804`. It does not activate virtual
geometry in ordinary scenes or change compatibility-renderer pixels.

## Instance and selection ABI

`GpuVirtualInstance` is exactly 208 bytes. Its original 128-byte
traversal-hot prefix remains current model, inverse-transpose normal rows, and
mesh/instance metadata. A 64-byte previous model and 16-byte tint are appended.
Constructors reject non-finite or non-affine current/previous transforms,
singular current transforms, non-finite tint, and caller-forged derived state.

The 32-byte selected-cluster ABI now carries the dense index into the exact
instance buffer uploaded for that dispatch. The caller-stable instance ID is
retained only in bounded page requests, where streaming feedback needs it.
This prevents sparse stable IDs from being interpreted as storage-buffer
offsets without growing selection records or adding a lookup pass.

## Atomic material remap

Cooked archives retain source glTF material slots; renderer materials use
generation-safe global IDs. `bind_mesh_materials` requires one unique mapping
for every source slot, including `None` for glTF's default material, and rejects
zero IDs. Duplicate, missing, unused, or zero mappings return before CPU or GPU
metadata changes. A valid binding writes all cluster material IDs before
publishing the mesh's `MATERIALS_BOUND` bit through the same ordered queue.

Traversal checks that bit before any counter reset, instance upload, parameter
upload, or compute dispatch. An unbound mesh therefore fails closed instead of
producing a plausible but incorrect fallback material. Rebinding is similarly
ordered behind older submitted draws and has no per-frame shader branch or
lookup cost.

## Real Metal oracle

The focused 24-test virtual-geometry suite passed. Its material/temporal oracle
uses one mesh with source material slots 3 and 9 remapped to global IDs 101 and
202. Two instances have sparse stable IDs 501 and 777 but dense GPU indices 0
and 1. Eight selected clusters preserve those dense indices and exact material
IDs. The production raw-page decoder reads 24 indexed corners from the real
pool/selection/instance buffers and proves:

- current world positions for translations 0 and 10;
- previous world positions for translations -2 and 8;
- distinct per-instance RGBA tints;
- inverse-transpose world normals;
- exact cluster/material and dense-instance identity.

The same test proves GPU cluster-table readback after a valid bind, unchanged
tables after three invalid bind classes, rejection of material ID zero, and
pre-dispatch rejection of an unbound mesh.

## Regression boundary and gates

The complete governed quick lane passed in 72 seconds:

- 403 shared unit tests passed and one existing hot-reload test was ignored;
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

No `Renderer`, `EngineState`, `ModelManager`, frame-graph, ordinary glTF, or FFI
owner changed. The shipping path constructs no new buffers, bindings, passes,
draws, or shader branches, so this checkpoint adds no default-path GPU memory,
CPU/GPU work, or pixel change.

## Remaining acceptance work

No #131 acceptance checkbox changes at this checkpoint. Production integration
still needs visibility-ID namespacing, raw virtual raster and exact PBR shading,
current/previous view-projection composition, every established MRT, a bounded
submission path without indirect-count support, conservative previous-frame
Hi-Z, asynchronous page feedback, crack/temporal corpora, 10-million-triangle
stress, total GPU timing, and integrated/discrete adapter qualification.
