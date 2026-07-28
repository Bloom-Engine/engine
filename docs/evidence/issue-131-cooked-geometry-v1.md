# Issue #131 cooked-geometry v1 evidence

This checkpoint establishes Bloom's opt-in offline meshlet/page artifact at
revision `59c8df7`. It does not enable a virtualized runtime path or change
accepted pixels.

## Static geometry proof

Two independent release cooks of
`examples/renderer-test/assets/DamagedHelmet.glb` are byte-identical:

- source: 15,452 eligible triangles;
- output: 258 leaf meshlets in 20 pages;
- page budget: 65,536 bytes;
- largest page: 63,216 bytes;
- payload: 1,230,128 bytes;
- complete artifact: 1,264,592 bytes;
- artifact SHA-256:
  `df089b23324fe8a8e00842a80b44894fd27b276c9124b8ff81dd77f8cf7b2cd2`;
- payload SHA-256:
  `67b0352a54d3d669d76d688f94899bcb8a665c561587123ea85d5d6a2b1061b0`.

The default 64-vertex/124-triangle leaf limits and 64 KiB page budget
therefore hold for a canonical textured glTF asset. Material identity and all
static shading vertex attributes remain in the artifact.

## Compatibility proof

`examples/test-gltf-watch/assets/Fox.glb` produces a valid 176-byte
metadata-only artifact:

- zero meshlets/pages/payload bytes;
- mesh 0, primitive 0 records the stable `skinned` reason;
- artifact SHA-256:
  `5e4125b6ba31649f5ebbd694d79242e19fdb0c561073beee08f490c12be063e2`.

The cooker does not reinterpret skinned geometry as static or silently omit
the reason. Equivalent records exist for morph targets, BLEND materials, and
non-triangle topology.

## Integrity and determinism gates

Thirteen release tests plus strict Clippy pass. They cover:

- deterministic triangle-order partitioning and byte-exact serialization;
- hard vertex/triangle/page limits and independently hashed pages;
- conservative AABB/sphere/normal-cone construction;
- deterministic missing-normal generation;
- real GLB buffer import without image decoding;
- metadata-only compatibility artifacts;
- atomic replacement of an existing artifact;
- rejection of bad magic/version/endian, non-canonical or overlapping ranges,
  truncated data, payload corruption, invalid indices, invalid hierarchy
  links, bad counts/stride/bounds, and NaN/Inf data.

The repository's quick `quality-contract` component runs format, Clippy, and
release tests for `bloom-cook`; it passed in one second from a warm build.
`geometry-inspect` validates the full file before reporting any records.

## Runtime neutrality

This commit changes only `tools/bloom-cook`, documentation, and its CI gate:

- zero production render passes or draws;
- zero runtime buffers, images, allocations, bindings, or shader branches;
- zero changes to ordinary glTF/immediate-mode dispatch;
- zero engine/runtime dependencies.

The current artifact contains leaf clusters only. Runtime loading, hierarchical
LOD construction, coarse fallback clusters, residency, GPU traversal, and
streaming remain later #131 milestones and must be qualified before any
end-state acceptance box is checked.

The detailed format and compatibility contract is in
`docs/virtualized-geometry.md`.
