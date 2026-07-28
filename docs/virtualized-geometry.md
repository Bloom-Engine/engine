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

# Opt in to the offline coarse hierarchy (up to 16 levels).
bloom-cook geometry scene.glb scene.bgeo --hierarchy-levels 8
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
The hierarchy level count is a cook setting and must therefore be captured by
the future #136 artifact key alongside the source hash and meshlet/page limits.

The reader rejects before payload access when it sees:

- unknown magic, format version, or endian tag;
- truncated, overlapping, gapped, non-canonical, or overflowing ranges;
- a file, payload, or page length mismatch;
- a payload or page hash mismatch;
- invalid vertex/index counts, stride, local indices, bounds, cone, hierarchy
  links, NaN/Inf values, or page-budget overflow;
- hierarchy roots without the coarse-root flag, missing or out-of-range parent
  groups, non-reciprocal group ranges, non-increasing levels,
  identity/material crossings, or decreasing accumulated error;
- unknown compatibility reason codes.

Writers run the strict reader on the in-memory result before atomically
installing it. This prevents a writer regression from putting an unchecked
artifact on disk.

## Geometry and compatibility behavior

By default, version 1 builds deterministic leaf meshlets in source triangle
order, exactly as the first milestone did. `--hierarchy-levels` is explicit
opt-in and uses meshoptimizer's deterministic locality-aware leaf builder
before constructing coarse levels. Bounds include object-space AABB/sphere and
a conservative face-normal cone. Double-sided material clusters explicitly
disable backface-cone rejection. Missing normals are regenerated
deterministically with area-weighted triangle normals. Opaque and alpha-masked
triangles are eligible.

## Atomic coarse hierarchy

Hierarchy traversal operates on atomic cluster groups. Every child stores the
first cluster and count of its replacement parent group. Every parent sibling
stores the same contiguous child range. A group can therefore simplify into
several ordinary 64-vertex meshlets; a future runtime must select all parents
or all children for that edge. The reader validates both directions before
accepting the archive.

Each level combines up to eight spatially ordered child groups and targets
half their triangle count. Exact complete-vertex welding restores adjacency
without merging normal, tangent, UV, or color discontinuities. The
attribute-aware simplifier weighs normals, tangents, both UV sets, and color,
and locks the complete group's topological border. Independently selected
neighboring groups therefore retain their shared boundary. A replacement that
does not reduce cluster count is rejected and its children become terminal
roots.

The simplifier reports absolute object-space error. Each parent group stores
that error plus the maximum error accumulated by its children. Parent
AABB/sphere bounds conservatively cover the complete replaced child group;
normal cones remain per rendered parent meshlet. The cook report exposes root
cluster/payload counts by level so an always-resident budget cannot be hidden
by aggregate totals.

The cooker integration uses the Rust `meshopt` wrapper and its vendored
meshoptimizer implementation under their permissive MIT/Apache-2.0 terms.

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

The default version 1 artifact remains byte-identical to the qualified
leaf-only milestone (`parent` and `first_child` absent, both relation counts
zero, level/error zero). Opt-in artifacts populate those formerly reserved
fields without changing the 128-byte cluster record or format version.
Runtime streaming remains disabled until residency, traversal, occlusion, and
fallback milestones are independently qualified.

## Qualification

The quick CI quality contract runs release tests and strict Clippy for
`bloom-cook`. The focused tests cover deterministic partitioning, hierarchy
construction and encoding, atomic reciprocal relations, monotonic
error/bounds, locked outer boundaries, limits, normal generation, conservative
bounds/cones, real GLB import, metadata-only compatibility artifacts, repeated
output replacement, and the corruption/range/hash cases above.

The canonical static smoke asset is
`examples/renderer-test/assets/DamagedHelmet.glb`. With default limits it
currently produces 15,452 triangles in 258 meshlets and 20 pages, with a
maximum page of 63,216 bytes under the 65,536-byte budget. Two independent
cooks are byte-identical. The canonical compatibility control is
`examples/test-gltf-watch/assets/Fox.glb`.
