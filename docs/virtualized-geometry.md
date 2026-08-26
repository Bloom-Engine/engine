# Virtualized geometry

Bloom's virtualized-geometry work is opt-in and staged. The #131 milestones now
cover the deterministic meshlet/page contract, strict runtime loading,
fixed-budget GPU residency, projected-error GPU hierarchy selection, raw-page
vertex decoding, bounded indirect draw emission, temporal/material ABI,
namespaced visibility raster, four-MRT PBR composition, and explicit production
`Renderer` ownership. Ordinary glTF rendering remains the default; this does
not yet claim complete Nanite-equivalent streaming, stress-scale, or
cross-backend qualification.

## Cook and inspect

```shell
cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  geometry scene.glb scene.bgeo

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  geometry-inspect scene.bgeo

# Content-addressed store and logical manifest (offline only).
cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  geometry-store scenes/example scene.glb out/assets \
  --hierarchy-levels 8 --vertex-format quantized32
```

The default leaf limits are 64 unique vertices and 124 triangles per meshlet.
They are cooker details, not public engine API. Qualification experiments may
override them:

```shell
bloom-cook geometry scene.glb scene.bgeo \
  --max-vertices 32 --max-triangles 48 --page-bytes 32768

# Opt in to the offline coarse hierarchy (up to 16 levels).
bloom-cook geometry scene.glb scene.bgeo --hierarchy-levels 8

# Opt in to the version 2 packed static-vertex payload.
bloom-cook geometry scene.glb scene.bgeo \
  --hierarchy-levels 8 --vertex-format quantized32
```

Vertex limits must be 3–255. Page size must be a power of two between 4 KiB
and 4 MiB. The default is 64 KiB. A meshlet that cannot fit in one page fails
the cook instead of silently exceeding the residency budget.

Both commands write machine-readable JSON. `geometry` reports source and
payload hashes, eligible triangle/meshlet/page counts, limits, maximum page
size, vertex encoding and measured reconstruction error, and every
compatibility-routed primitive. For a packed cook it also constructs the
equivalent float32 artifact in memory and reports exact payload/root-page byte
reductions. `geometry-inspect` validates the complete artifact before
reporting it.

`geometry-store` uses the same cooker but writes the immutable artifact under
its complete SHA-256 and atomically maps a logical ID to it. Its recipe key
includes the source closure, hierarchy level count, meshlet/page limits,
vertex format, and explicit recipe version. A valid repeat is a strict
zero-write cache hit. See `docs/cooked-asset-store.md`. Runtime code can now
consume the selected artifact bytes plus the identity fields returned by that
index; asynchronous store lookup remains a later integration milestone.

## Versioned payload contract

`.bgeo` version 1 is little-endian and uses these fixed tables:

| Record | Size | Purpose |
|---|---:|---|
| Header | 160 B | Magic/version/endian tag, counts, canonical offsets, source and payload SHA-256, page budget |
| Cluster | 128 B | Mesh/primitive/material identity, page and payload ranges, bounds, normal cone, error and hierarchy links |
| Page | 64 B | Contiguous payload/cluster range for one LOD/residency class, independent SHA-256 |
| Compatibility | 16 B | Mesh/primitive, stable reason code, reason detail |

Cluster payloads contain fixed 72-byte static vertices—position, normal,
tangent, UV0, UV1, and color—followed by three local `u8` indices per
triangle. Clusters do not cross pages. Material boundaries cannot cross
clusters because each glTF primitive is partitioned independently.

Version 2 is an explicit `--vertex-format quantized32` opt-in. It keeps the
same endian-defined tables, hierarchy, page limits, hashes, and local indices,
but uses this fixed 32-byte vertex layout:

| Byte range | Encoding |
|---|---|
| 0–5 | Cluster-AABB-local position, three `UNORM16` components |
| 6–9 | Unit normal, octahedral `SNORM16x2` |
| 10–13 | Unit tangent direction, octahedral `SNORM16x2` |
| 14–21 | UV0 and UV1, four finite IEEE 754 binary16 values |
| 22–25 | Vertex color, `UNORM8x4` |
| 26–27 | Tangent handedness, `SNORM16` |
| 28–29 | Tangent-valid flags |
| 30–31 | Reserved zero padding |

The tangent-valid bit preserves an imported all-zero missing-tangent sentinel;
the decoder does not manufacture a direction that could silently enable
normal mapping. Values outside the representable safety contract—non-finite
components, UVs outside finite binary16, colors outside 0–1, invalid
directions or handedness—fail the packed cook. Unknown flags, non-zero
padding, non-finite half values, or a malformed sentinel fail strict
inspection.

Packed payloads are not selected automatically. The lossless version 1 path
remains the default and is byte-identical to the previously qualified output.
This allows the future asset policy to enforce measured error limits per
asset; in particular, large tiled UV ranges must be evaluated from the
reported absolute error instead of assuming binary16 is always invisible.

The source hash covers the glTF/GLB bytes and the complete resolved buffer
contents, including external buffers. The payload and every page have separate
SHA-256 hashes. Regenerating the same source and settings is byte-identical.
The hierarchy level count and vertex format are cook settings and must
therefore be captured by the future #136 artifact key alongside the source
hash and meshlet/page limits.

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
triangles are cooked into static clusters. Runtime ownership currently retains
alpha-masked primitives on the ordinary renderer because virtual visibility
does not yet own their exact texture, sampler, and cutoff test.

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

Before encoding, hierarchy clusters are deterministically remapped into
coarse-first order while every atomic relation range remains contiguous.
Coarse roots occupy a page prefix, and no page may mix root/streamable classes
or LOD levels. `geometry` and `geometry-inspect` report the exact root-page
count and bytes. The reader rejects mixed classes or roots placed after
streamable pages, making the future always-resident upload a bounded prefix
instead of a heuristic scan.

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

## Runtime loading and residency boundary

The `models3d` feature now exposes `bloom_shared::virtual_geometry` as an
explicit opt-in. `VirtualGeometryAsset::from_bytes` validates the full archive
before returning immutable metadata or page slices. The indexed form also
checks the complete file length/hash, format version, payload hash, and source
hash against #136's selected artifact identity. The reader lives in the small
`bloom-geometry-format` crate and is shared by the cooker and runtime, so the
writer cannot validate against a different interpretation than a backend will
load.

`VirtualGeometryResidency` models a hard GPU page-cache budget. Construction
pins the validated coarse-root page prefix and fails if the roots alone exceed
the budget. Group requests are atomic: every page needed by an atomic cluster
group fits or residency is unchanged. Streamable pages use deterministic LRU
eviction with page index as the tie-break. A missing detail group walks only
validated parent-group links until it finds a completely resident ancestor;
partially resident groups are never rendered. Telemetry reports pinned,
resident, upload, eviction, exact-resolution, fallback, and unresolved counts.

`GpuVirtualGeometryPool` is the corresponding explicit GPU owner. It uses one
fixed-size raw-page storage buffer rather than the compatibility renderer's
growable, expanded `Vertex3D`/`u32` arena. Its configuration fixes the physical
bytes, slot stride, mesh-table records, page-table records, geometry upload
bytes/pages per frame, and evictions per frame before any allocation occurs.
Construction fails against device buffer/binding limits instead of silently
shrinking a requested budget.

Registration allocates a generational `VirtualMeshId`, a contiguous logical
page-table range, and physical slots for every validated coarse-root page.
`VirtualPageId` combines that mesh generation with a local page index, so a
retired page cannot alias a later mesh that reuses the same descriptor slot.
The GPU mesh and page entries carry the complete mesh ID; a zero physical slot
means missing. Retired IDs, page-table ranges, and physical slots are not
reused until a queue-completion callback proves older commands have finished.

Detail requests plan every upload and deterministic global-LRU eviction before
writing anything. A complete atomic cluster group fits both the hard physical
pool and the remaining per-frame upload/eviction limits, or no page mapping is
changed. The physical buffer stores only bytes returned by the fully validated
archive reader. GPU-visible entries expose slot, payload length, ownership,
resident state, and pinned state. Telemetry distinguishes allocated GPU bytes,
resident slot bytes, useful payload bytes, pinned/retiring slots, frame and
lifetime upload/eviction counts, denials, and exact/ancestor fallback results.

`GpuVirtualHierarchySelector` consumes fixed 208-byte instance records and
walks each atomic hierarchy group on the GPU. The traversal-hot 128-byte prefix
holds the current model transform, derived normal rows, mesh/instance identity,
and cone-safety flag. The appended render state holds the previous model
transform and current tint. Projected pixel error controls LOD. Frustum and
transform-safe normal-cone tests reject invisible work; non-uniform or sheared
transforms conservatively disable only cone rejection. Refinement requires the
complete child group to be resident. Otherwise the selector keeps the resident
ancestor and writes bounded page requests. Camera cuts and instance motion
consume no prior selection state.

Selected-cluster records address instances by their dense dispatch index, not
the caller's stable instance ID. This keeps vertex pulling exact when stable IDs
are sparse. Page requests retain the stable ID for asynchronous feedback.
Before traversal, material binding must atomically map every archive material
slot—including the glTF default slot—to a nonzero generation-safe renderer
material ID. Production model callers use `bind_model_virtual_materials`, which
verifies the model/archive source closure, derives exact base-PBR records from
the loaded glTF meshes and renderer texture IDs, owns those IDs until rebind or
virtual shutdown, and rejects transmission/layered materials that are not yet
virtual-authoritative. The lower-level `bind_mesh_materials` remains available
for procedural and tooling assets. Duplicate, missing, unused, conflicting, or
zero bindings change no CPU or GPU table. Traversal rejects an unbound mesh
before dispatch, so streamed geometry cannot silently shade through material
zero.

`GpuVirtualDrawEmitter` converts the bounded selected table into compact
16-byte non-indexed indirect commands. `first_instance` addresses the matching
selected-cluster record for vertex pulling. Selection overflow, an invalid
record, or a missing current page publishes a zero draw count for the whole
virtual batch; request overflow remains safe because the complete resident
ancestor batch is still selected. Construction validates command-buffer and
compute-dispatch limits up front.

The shared WGSL page decoder reads local `u8` triangle indices directly from
the fixed physical pool and reconstructs every version 1 Float32 or version 2
quantized vertex lane. A Metal compute oracle reads the real mesh, cluster,
selection, instance, and physical-page buffers. It proves current/previous
world positions, inverse-transpose world normals, tint, and remapped materials
for sparse caller IDs and dense GPU instance addresses. A separate Metal render
oracle consumes the emitted indirect commands and proves the exact
`first_instance` values. An eventual production consumer must still combine
the temporal models with the frame's current and previous view-projection
state, use indirect-count support or provide a separately qualified bounded
fallback, and reproduce every visibility MRT. Unsupported adapters remain on
the compatibility renderer.

`GpuVirtualVisibilityRaster` is the first raw visibility consumer. It pulls
the emitted triangles from the physical page, cluster, render-ready selection,
and instance buffers and writes Bloom's shared `Rg32Uint` target plus depth. Draw
word bit 31 selects the virtual namespace; the lower 31 bits address the
selected record. Compatibility IDs keep bit 31 clear, while `0xffffffff`
remains the unambiguous background sentinel. Primitive word bit 31 continues
to carry front-face orientation independently.

Each unchanged 32-byte selected record carries an absolute cluster-table index,
an absolute physical-page byte base, and packed vertex encoding. The cluster's
formerly reserved payload lane carries its generation-safe owning mesh ID.
Together those fields reject cross-mesh aliases and remove one mesh-table fetch
and one storage binding from every raster and reconstruction invocation.

Construction requires `PRIMITIVE_INDEX`, `INDIRECT_FIRST_INSTANCE`, four
vertex-stage storage buffers, and a draw capacity that fits the namespace.
Adapters with `MULTI_DRAW_INDIRECT_COUNT` use the exact counted stream. Other
qualified adapters use the bounded 22-bin GPU compaction path, whose submitted
vertex amplification is strictly below 2x and whose empty bins carry zero
instances. Alpha-masked clusters are discarded and remain compatibility-owned;
single-sided clusters reject back faces while double-sided clusters preserve
the face bit for later shading. The 128-byte frame record already carries
current and previous view-projection transforms, although this raster consumes
only the current transform.

The renderer integration remains deliberately explicit:

- zero production render passes, draws, buffers, bindings, allocations, or
  shader branches unless a caller explicitly constructs the GPU pool;
- zero changes to existing immediate-mode or glTF selection and pixels;
- no silent `.bgeo` replacement of an ordinary model;
- no asynchronous file/store IO; validated archive bytes remain memory-owned
  while GPU request feedback and fixed-budget uploads run asynchronously.

The opt-in virtual PBR consumer reconstructs perspective-correct current and
previous clip positions, inverse-transpose normals, mirrored tangent
handedness, UVs, vertex tint, remapped material identity, and face state before
calling the authoritative scene material evaluator. Its production pipeline
uses the established four MRTs and fits the renderer's eight fragment-stage
storage-buffer contract. `Renderer::enable_virtual_geometry` is the explicit
attachment point, so ordinary frames still construct and draw none of this
work. Normal callers use `submit_virtual_geometry_current_view`: hierarchy
selection receives the renderer-owned unjittered frustum/projection scale while
visibility and velocity receive the exact current/previous jittered transforms.
The fully explicit view form remains available for offline tools.

Runtime model routing validates the complete glTF source closure against the
archive, preserves source mesh/primitive/placement identity, and rejects an
incomplete or multiply owned partition before submission. A filtered virtual
instance traverses only its source glTF mesh within a shared scene archive.
Cooked compatibility records, plus alpha-masked primitives deferred by the
visibility contract, enter a compact compatibility-only cache and submit
allocation-free through the ordinary renderer. Virtual-eligible vertex/index
payloads are not uploaded to the ordinary static arena. This prevents both a
per-placement instance from duplicating every mesh in a scene archive and the
compatibility bridge from quietly duplicating the complete model on the GPU.

Missing-page feedback uses two fixed MAP_READ buffers and never waits in the
render loop. Each completed traversal copies at most 4,096 request records by
default (clamped to the traversal capacity), rejects out-of-order camera
completions, canonicalizes repeated instances to one mesh/group request, and
retains at most 8,192 pending groups. Newest feedback is serviced first;
generation-stale or malformed requests are discarded. Atomic group uploads
continue to obey the pool's existing per-frame byte, page, and eviction limits,
and budget-blocked groups stay pending while the nearest resident ancestor
remains visible. Advanced callers can override these feedback limits through
`enable_virtual_geometry_with_streaming`; ordinary rendering allocates and
records none of the feedback path.

An opt-in virtual submission also captures a private 256x256 previous-frame
max-depth pyramid (349,524 texture bytes) after the renderer's current linear
depth build. The next traversal can reject an atomic hierarchy group only when
every in-frustum cluster is proven behind the farthest depth over its complete
screen footprint. Camera cuts, resize, skipped frames, new instances,
near-plane/off-screen bounds, or motion beyond one base-grid cell fail open.
Queries union previous/current bounds, expand by two cells, and include
relative and absolute depth bias. Consequently an uncertain group remains
visible instead of risking a hole. Nine separate R32Float textures avoid
Metal's sampled/written mip hazard. Ordinary frames add no pass or allocation,
and virtual-idle frames add no pass. Asynchronous feedback telemetry exposes
the latest visible, frustum-culled, occlusion-culled, and
occlusion-uncertain group counts without synchronizing the render loop.

The next #131 runtime milestones are detailed-Bistro camera-motion/pixel
qualification, a 10M-source-triangle residency stress, asynchronous #136
store/index IO behind the GPU feedback boundary, and cross-backend timing. The
compatibility renderer remains responsible for unsupported and not-yet-qualified
content throughout that work.

The default version 1 artifact remains byte-identical to the qualified
leaf-only milestone (`parent` and `first_child` absent, both relation counts
zero, level/error zero). Opt-in hierarchy artifacts populate those formerly
reserved fields without changing the 128-byte cluster record. Opt-in packed
vertices use format version 2 so a version 1 reader can never reinterpret the
32-byte stride as float32. Runtime activation remains explicit opt-in until
store-backed IO, virtual occlusion stress/motion, and platform milestones are
independently qualified.

## Qualification

The quick CI quality contract runs release tests and strict Clippy for
`bloom-cook`. The focused tests cover deterministic partitioning, hierarchy
construction and encoding, atomic reciprocal relations, monotonic
error/bounds, locked outer boundaries, packed reconstruction/error limits,
missing-tangent preservation, non-canonical packed-bit rejection, limits,
normal generation, conservative bounds/cones, real GLB import, metadata-only
compatibility artifacts, repeated output replacement, and the
corruption/range/hash cases above.

The canonical static smoke asset is
`examples/renderer-test/assets/DamagedHelmet.glb`. With default limits it
currently produces 15,452 triangles in 258 meshlets and 20 pages, with a
maximum page of 63,216 bytes under the 65,536-byte budget. Two independent
cooks are byte-identical. The canonical compatibility control is
`examples/test-gltf-watch/assets/Fox.glb`.

With `--hierarchy-levels 8`, the same static asset produces 247 leaf clusters,
385 parent clusters, and 100 level-3 coarse roots. The roots reduce the
triangle set by 70.8% and raw payload by 60.2%; two cooks are byte-identical.
Those roots occupy the first eight pages and require exactly 469,360 resident
payload bytes, only 790 bytes of packing overhead beyond their raw cluster
payload. This is structural hierarchy/page qualification, not yet a measured
runtime residency budget.
The quantized version 2 hierarchy reduces the same payload from 2,937,040 to
1,363,968 bytes (53.56%) and the root-page prefix from 469,360 to 216,544
bytes (53.86%). Its two independent artifacts are byte-identical; maximum
Damaged Helmet reconstruction errors are 0.00001532 object units for position,
0.00342 degrees for normals, and 0.0004883 for UVs. Sponza separately
qualifies non-zero tangents at 0.00361 degrees and exposes its 0.01381
large-range UV error in the cook report.

The hierarchy, page-placement, and packed-payload records are
`docs/evidence/issue-131-atomic-hierarchy-v1.{md,json}` and
`docs/evidence/issue-131-coarse-page-prefix-v1.{md,json}`, and
`docs/evidence/issue-131-quantized-vertices-v2.{md,json}`.
The content-addressed manifest handoff to #136 is recorded in
`docs/evidence/issue-136-geometry-store-v1.{md,json}`; its deterministic
single-table lookup handoff is in
`docs/evidence/issue-136-asset-index-v1.{md,json}`.
Explicit platform/quality variants, ordered fallback, and the Bistro
loose-store qualification are recorded in
`docs/evidence/issue-136-asset-variants-v2.{md,json}`.
The fixed physical GPU pool, stable ID/page-table ABI, bounded frame work, and
Metal readback proof are recorded in
`docs/evidence/issue-131-gpu-page-pool-v1.{md,json}`.
GPU hierarchy selection and independent CPU parity are recorded in
`docs/evidence/issue-131-gpu-hierarchy-traversal-v1.{md,json}`. Raw-page decode,
bounded indirect emission, and executable Metal raster proof are recorded in
`docs/evidence/issue-131-virtual-draw-emission-v1.{md,json}`. Collision-free
visibility namespacing and raw virtual ID/depth rasterization are recorded in
`docs/evidence/issue-131-virtual-visibility-raster-v1.{md,json}`.
