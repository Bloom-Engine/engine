# Issue #131 cooked compatibility routing v1 evidence

This checkpoint qualifies exact cooked/runtime ownership routing at revision
`88e8d048f4af0e738043476845dfd779125e71cb`. It prevents a scene-level
`.bgeo` archive from being traversed in full for every glTF node placement and
keeps every not-yet-virtualized primitive on Bloom's established cached model
renderer. It does not claim complete #131 activation.

## Exact source ownership

The cooker and runtime now share one versioned source-closure hash over the
source glTF/GLB bytes and every resolved buffer in glTF buffer-index order.
Runtime routing fails closed when that closure is missing or differs from the
archive. The model loaders retain source mesh, primitive, and node-placement
identity without duplicating immutable vertex/index payloads.

Every runtime primitive must resolve to exactly one archive partition. Archive
decode rejects duplicate compatibility records and any compatibility record
that overlaps a clustered primitive. Runtime routing also rejects incomplete
placement metadata, a primitive absent from either partition, or an archive
primitive absent from the model.

## Mixed-scene submission

One virtual instance is emitted per source-mesh node placement, with current
and previous outer transforms composed with the authored node transform. A
source-mesh filter is carried in the existing 208-byte instance record and is
applied by both CPU-reference and GPU hierarchy traversal. Unfiltered traversal
of a multi-source archive is rejected before dispatch.

Cooked compatibility records are submitted through a canonical, allocation-free
subset of the ordinary cached model renderer. The ordinary full-model method
is unchanged. The subset preflight rejects absent caches and out-of-range,
duplicate, or unordered indices before recording any uniforms or draw commands.
The full transform, material flags, motion history, authored node transforms,
and tint follow the established renderer path.

Alpha-masked clusters remain ordinary-renderer owned until virtual visibility
can evaluate the exact material texture, sampler, and cutoff contract. This
deliberately trades redundant selection work in a mixed source mesh for correct
coverage and no holes.

## Real Metal routing oracle

The shared archive fixture contains two independent source-mesh hierarchies.
The production GPU selector chose cluster-table records `[4, 5]` for source
mesh 0 and `[6, 7]` for source mesh 1. Submitting both placements selected
exactly four records with the corresponding dense instance indices. The GPU
result matched the CPU reference, produced no missing-page requests, and
rejected both an unfiltered multi-source instance and a compatibility-only
source mesh before dispatch.

The model route tests cover repeated node placements with mixed virtual and
compatibility primitives, current/previous transform composition, stable
instance IDs, source-closure mismatch, incomplete/unrouted content, and
alpha-mask fallback ownership.

## Qualification

The exact committed tree passed:

- 454 shared library tests, with one existing ignored test;
- the complete real-GPU golden corpus: 77 passed and two hardware-specific
  tests ignored;
- 30 cooker tests, including strict overlap/duplicate partition rejection;
- native no-default and WebAssembly `web,models3d` checks;
- strict format/diff checks and strict cooker/geometry-format Clippy;
- the shared correctness/suspicious/performance Clippy run produced no
  diagnostic in this slice, but remains red on eight pre-existing findings in
  `input.rs`, `renderer/mod.rs`, `string_header.rs`, and `shadows.rs`.

The repository file-size ratchet remains red only for three pre-existing files
outside this slice: `renderer/shaders/post.rs`, `renderer/shaders/ssgi.rs`, and
`golden_render/temporal_history.rs`. No file introduced or changed by this
checkpoint exceeds its governed ceiling.

## Remaining acceptance work

No #131 acceptance checkbox changes at this checkpoint. Production acceptance
still needs bounded asynchronous missing-page feedback/streaming, conservative
previous-frame virtual Hi-Z, approved Bistro motion/parity qualification, a
10-million-source-triangle stress asset, and integrated/discrete Metal, Vulkan,
and Direct3D 12 timing and quality evidence.
