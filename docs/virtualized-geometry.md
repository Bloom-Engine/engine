# Virtualized geometry

Bloom's virtualized-geometry work is opt-in and staged. The first #131
milestone establishes a deterministic offline meshlet/page contract. It does
not enable a new runtime renderer, change ordinary glTF loading, or claim
Nanite-equivalent hierarchy/streaming.

## Cook and inspect

```shell
cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  geometry scene.glb scene.bgeo

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  geometry-inspect scene.bgeo
```

The default leaf limits are 64 unique vertices and 124 triangles per meshlet.
They are cooker details, not public engine API. Qualification experiments may
override them:

```shell
bloom-cook geometry scene.glb scene.bgeo \
  --max-vertices 32 --max-triangles 48 --page-bytes 32768
```

Vertex limits must be 3–255. Page size must be a power of two between 4 KiB
and 4 MiB. The default is 64 KiB. A meshlet that cannot fit in one page fails
the cook instead of silently exceeding the residency budget.

Both commands write machine-readable JSON. `geometry` reports source and
payload hashes, eligible triangle/meshlet/page counts, limits, maximum page
size, and every compatibility-routed primitive. `geometry-inspect` validates
the complete artifact before reporting it.

## Version 1 contract

`.bgeo` version 1 is little-endian and uses these fixed tables:

| Record | Size | Purpose |
|---|---:|---|
| Header | 160 B | Magic/version/endian tag, counts, canonical offsets, source and payload SHA-256, page budget |
| Cluster | 128 B | Mesh/primitive/material identity, page and payload ranges, bounds, normal cone, error and hierarchy links |
| Page | 64 B | Contiguous payload range, contiguous cluster range, independent SHA-256 |
| Compatibility | 16 B | Mesh/primitive, stable reason code, reason detail |

Cluster payloads contain fixed 72-byte static vertices—position, normal,
tangent, UV0, UV1, and color—followed by three local `u8` indices per
triangle. Clusters do not cross pages. Material boundaries cannot cross
clusters because each glTF primitive is partitioned independently.

The source hash covers the glTF/GLB bytes and the complete resolved buffer
contents, including external buffers. The payload and every page have separate
SHA-256 hashes. Regenerating the same source and settings is byte-identical.

The reader rejects before payload access when it sees:

- unknown magic, format version, or endian tag;
- truncated, overlapping, gapped, non-canonical, or overflowing ranges;
- a file, payload, or page length mismatch;
- a payload or page hash mismatch;
- invalid vertex/index counts, stride, local indices, bounds, cone, hierarchy
  links, NaN/Inf values, or page-budget overflow;
- unknown compatibility reason codes.

Writers run the strict reader on the in-memory result before atomically
installing it. This prevents a writer regression from putting an unchecked
artifact on disk.

## Geometry and compatibility behavior

Version 1 builds deterministic leaf meshlets in source triangle order. Bounds
include object-space AABB/sphere and a conservative face-normal cone.
Double-sided material clusters explicitly disable backface-cone rejection.
Missing normals are regenerated deterministically with area-weighted triangle
normals. Opaque and alpha-masked triangles are eligible.

The following content is recorded for the existing compatibility renderer and
does not produce virtualized meshlets:

| glTF content | Compatibility reason |
|---|---|
| Points, lines, strips, or fans | `non-triangle-topology` |
| A mesh referenced by a skinned node | `skinned` |
| A primitive with morph targets | `morph-targets` |
| `alphaMode: BLEND` | `alpha-blend` |

An entirely incompatible asset still produces a valid metadata-only `.bgeo`.
For example, Fox records its skinned primitive and zero pages. This makes the
fallback decision inspectable instead of silently treating unsupported
geometry as static.

## Runtime and performance boundary

This milestone is cooker-only:

- zero production render passes, draws, buffers, bindings, allocations, or
  shader branches;
- zero changes to existing immediate-mode or glTF pixels;
- no runtime `.bgeo` selection or silent fallback;
- no new dependency in the engine/runtime crates.

Runtime integration remains gated on the #131 dependencies:

- #136 must own the versioned asset database/provenance and asynchronous
  artifact lookup;
- #27 must demonstrate a measured net-positive visibility/material path
  before virtualized geometry targets it;
- the existing #28 shared geometry arena and indirect submission remain the
  compatibility/performance foundation.

The format reserves parent, first-child, child-count, and geometric-error
fields, but version 1 currently emits leaf-only clusters (`parent` and
`first_child` are absent, `child_count = 0`, `error = 0`). A later milestone
must build and qualify the coarse always-resident hierarchy before runtime
streaming is enabled. Until then, no #131 end-state acceptance box is complete.

## Qualification

The quick CI quality contract runs release tests and strict Clippy for
`bloom-cook`. The focused tests cover deterministic partitioning and encoding,
limits, normal generation, conservative bounds/cones, real GLB import,
metadata-only compatibility artifacts, repeated output replacement, and the
corruption/range/hash cases above.

The canonical static smoke asset is
`examples/renderer-test/assets/DamagedHelmet.glb`. With default limits it
currently produces 15,452 triangles in 258 meshlets and 20 pages, with a
maximum page of 63,216 bytes under the 65,536-byte budget. Two independent
cooks are byte-identical. The canonical compatibility control is
`examples/test-gltf-watch/assets/Fox.glb`.
