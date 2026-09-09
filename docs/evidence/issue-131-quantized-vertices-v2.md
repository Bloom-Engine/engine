# Issue #131 quantized static vertices v2 evidence

This checkpoint qualifies a deterministic, opt-in packed static-vertex
payload at revision `2c51d097583bb33bfef4ff4803c7e2570927bc1b`.
It reduces future storage, IO, residency, and vertex-fetch demand without
changing the shipping renderer.

## Format and safety contract

`--vertex-format quantized32` emits `.bgeo` version 2 with a fixed 32-byte
vertex:

- cluster-AABB-local `UNORM16x3` position;
- octahedral `SNORM16x2` normal and tangent direction;
- finite binary16 UV0 and UV1;
- `UNORM8x4` color;
- `SNORM16` tangent handedness;
- an explicit tangent-valid bit and reserved-zero padding.

The validity bit preserves the existing all-zero missing-tangent sentinel, so
packing cannot silently enable a tangent-space path. The writer rejects values
that cannot be represented safely. The strict reader rejects the wrong
version/stride, unknown flags, non-zero padding, non-finite half values,
malformed tangent sentinels, invalid ranges, and all pre-existing
hash/page/hierarchy failures before exposing the archive.

Version 1 float32 remains the default. Two established controls retain their
exact hashes:

- leaf-only:
  `df089b23324fe8a8e00842a80b44894fd27b276c9124b8ff81dd77f8cf7b2cd2`;
- eight-level hierarchy:
  `da0f68d731ca42a95a54e8ff157a62315a71dfa5c78d62b2bccd8ec59c0124f7`.

## Canonical size and reconstruction proof

Two independent release cooks of Damaged Helmet with
`--hierarchy-levels 8 --vertex-format quantized32` are byte-identical:

- 632 clusters / 33,706 hierarchy triangles;
- 2,937,040-byte float32 payload becomes 1,363,968 bytes, a
  1,573,072-byte (53.56%) reduction;
- 48 pages become 23, with a 65,504-byte maximum under the 65,536-byte budget;
- the root prefix becomes 4 pages / 216,544 bytes versus 469,360 bytes
  float32, a 53.86% reduction;
- complete artifact: 1,446,496 bytes;
- artifact SHA-256:
  `6c8f924e8dad74a5acd8e9acfb21795b153559cedb880862eb0a7f23eee5bc62`;
- payload SHA-256:
  `3450ff9074ddee3adda5d1373ddbe17d9a492c7130f8cae41a965dc4a2486169`.

Maximum reconstruction errors measured by decoding the written artifact:

- position: 0.0000153184 object units, or 0.0000082401 of a cluster extent;
- normal: 0.0034164 degrees;
- UV: 0.0004882813;
- color and tangent handedness: zero on this asset.

Damaged Helmet has no source tangents. A separate Sponza hierarchy exercises
the complete tangent path over 7,947 clusters: maximum tangent error is
0.0036042 degrees and normal error is 0.0035862 degrees. Its large tiled UV
ranges produce a reported 0.0138016 maximum binary16 error. This value is
recorded rather than hidden: the packed format remains opt-in and a future
asset-policy milestone must decide acceptable UV error per asset.

## Qualification and runtime boundary

- 21 release tests, formatting, strict Clippy, and the file-line gate pass.
- Tests cover bounded reconstruction, deterministic output, degenerate
  position axes, missing tangents, unsafe source values, non-canonical packed
  bits, and non-finite packed UVs.
- `geometry-inspect` strictly validates the final artifact.
- No runtime crate, render pass, draw, buffer, binding, shader branch, or
  accepted pixel changes.

This does not check a runtime acceptance box on #131. Asset-database keying,
runtime decode/upload, residency, traversal, fallback, and visual/performance
qualification remain gated on #136, #27, and the later #131 milestones.
