# Issue #131 atomic hierarchy v1 evidence

This checkpoint qualifies the opt-in offline hierarchy implemented at revision
`845ff9cbd535c54c85a4bcfd291427090f4bc911`. It does not enable runtime
selection, residency, streaming, or rendering.

## Hierarchy contract

- A hierarchy edge atomically replaces one contiguous child range with one
  contiguous parent range. It never forces a child group into one oversized
  meshlet.
- All parents in a replacement group share the child range. All children
  share the parent range. The strict reader validates both directions,
  increasing LOD levels, non-decreasing accumulated error, identity/material
  boundaries, root flags, and table bounds.
- Leaf and parent meshlets keep the configured 64-vertex/124-triangle default
  limits and remain independently pageable.
- Exact complete-vertex welding does not merge normal, tangent, UV, or color
  discontinuities.
- Attribute-aware simplification includes normals, tangents, UV0, UV1, and
  color. The complete group border is locked so independently selected
  neighboring groups retain their shared boundary.
- Parent errors add the simplifier's absolute object-space result to the
  maximum accumulated child error. Parent AABB/sphere bounds cover the full
  child group.
- A replacement that does not reduce cluster count is discarded. Its children
  become explicit coarse roots rather than adding a duplicate-only level.

## Deterministic static-asset proof

Two independent release cooks of
`examples/renderer-test/assets/DamagedHelmet.glb` with
`--hierarchy-levels 8` are byte-identical:

- 15,452 source triangles in 247 locality-aware leaf clusters;
- 385 parent clusters and 632 clusters across all levels;
- maximum generated level 3;
- 100 terminal coarse roots, all at level 3;
- coarse roots contain 4,510 triangles, a 70.812840% reduction from the leaf
  triangle set;
- coarse-root raw payload is 468,570 bytes, a 60.183646% reduction from the
  1,176,828-byte leaf payload;
- maximum accumulated absolute object-space error:
  `2.467940092086792`;
- 47 pages, each within the 65,536-byte hard budget;
- complete artifact: 3,021,104 bytes;
- artifact SHA-256:
  `57f8cc200da84c7ab63cdcd7bf02cb0c8336a96777a2ccb7fc49beaa0af3fd10`;
- payload SHA-256:
  `3546847f428431bd15681e32d451561744d5f5162729eb4c9cbeb35dc3b12915`.

The root set is deliberately reported, not declared a final runtime residency
budget. Quantized/compressed vertex payloads, page placement, traversal, and
measured residency remain separate gates.

## Regression and compatibility proof

The same Damaged Helmet cook without `--hierarchy-levels` remains byte-for-byte
identical to the first qualified leaf milestone:

- 258 leaf meshlets, 20 pages, and 1,230,128 payload bytes;
- artifact SHA-256:
  `df089b23324fe8a8e00842a80b44894fd27b276c9124b8ff81dd77f8cf7b2cd2`;
- payload SHA-256:
  `67b0352a54d3d669d76d688f94899bcb8a665c561587123ea85d5d6a2b1061b0`.

The hierarchy-enabled Fox control also remains the exact 176-byte,
metadata-only `skinned` compatibility artifact with SHA-256
`5e4125b6ba31649f5ebbd694d79242e19fdb0c561073beee08f490c12be063e2`.
Unsupported geometry is not silently treated as static.

## Automated gates and runtime neutrality

Sixteen release tests, formatting, and strict Clippy pass. The focused
hierarchy tests cover byte determinism, reciprocal atomic groups, monotonic
levels/error/bounds, locked outer-boundary preservation, and rejection of a
corrupted parent-group count.

This checkpoint changes only the offline `bloom-cook` tool and documentation:

- zero production render passes, draws, buffers, images, allocations,
  bindings, or shader branches;
- zero engine/runtime dependencies;
- zero changes to current renderer pixels or frame cost.

The next #131 work is runtime-independent payload quantization/compression and
page/root placement qualification, followed by #136 artifact integration.
GPU traversal remains gated on a measured net-positive #27 path.
