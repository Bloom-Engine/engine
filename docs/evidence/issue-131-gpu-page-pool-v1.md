# Issue #131 fixed GPU page pool v1 evidence

This checkpoint qualifies Bloom's first real virtual-geometry GPU residency
owner at revision `6dbbb02b48a777e6dd3f33fd3e15add1807626ac`. It does not yet
select clusters on the GPU or replace ordinary model submission.

## Ownership and ABI

`GpuVirtualGeometryPool` is constructed explicitly under `models3d`; no
`Renderer`, `EngineState`, `ModelManager`, frame-graph, shader, or FFI owner was
added. The existing #28 arena remains the expanded `Vertex3D`/`u32`
compatibility path. The new pool instead allocates three fixed storage buffers:

- compact cooked page bytes in fixed-stride physical slots;
- 32-byte generational virtual-mesh records;
- 16-byte logical-page records containing physical slot, bounded payload
  length, complete mesh ID, and resident/pinned flags.

`VirtualMeshId` follows Bloom's established 20-bit one-based slot / 12-bit
generation convention. `VirtualPageId` combines that generation-safe mesh ID
with a local page index. Zero remains the shader fallback/missing value. A
retired ID, its page-table range, and its physical slots remain quarantined
until `Queue::on_submitted_work_done` proves prior GPU references completed.

Configuration fixes physical bytes, page stride, mesh/page record counts,
geometry upload bytes/pages per frame, and evictions per frame before
allocation. Invalid alignment/ranges and resources exceeding either the
device's buffer-size or storage-binding-size limit fail before allocation.

## Metal readback proof

The release oracle ran on the Apple M1 Max 32-core GPU through wgpu/Metal. Its
validated three-level fixture uses one 448-byte root page, two 224-byte middle
pages, and two 448-byte leaf pages. The test configuration is deliberately
small enough to force replacement:

| Resource | Exact bytes |
|---|---:|
| Physical pool: 3 × 4 KiB slots | 12,288 |
| Page table: 16 × 16-byte records | 256 |
| Mesh table: 2 × 32-byte records | 64 |
| Total fixed GPU ownership | 12,608 |

The 448-byte coarse root occupies one pinned 4 KiB slot. After a fresh frame,
three detail pages totaling 896 bytes upload; the third replaces the globally
least-recently-used non-pinned page. Final physical residency is exactly all
three slots (12,288 bytes), while useful validated payload is exactly 1,120
bytes. The requested leaf group resolves at LOD 0.

The test copies all three buffers to `MAP_READ` staging buffers. It proves:

- the physical slot contains the exact validated archive page bytes;
- the GPU page-table record is byte-identical to its CPU mirror;
- the GPU mesh-table record is byte-identical to its CPU mirror;
- allocated buffer sizes and telemetry equal the configured byte counts.

No shader interprets these bytes yet, so this proof covers ownership,
addressing, transfer, and budget enforcement rather than rendered pixels.

## Atomicity and bounded work

Every group transition is fully planned before queue writes. Required pages
are protected while free slots and then global LRU candidates are selected by
`(last_use, physical_slot)`; pinned and GPU-retiring slots are never candidates.

Three independent negative paths pass:

- a two-page atomic leaf group is rejected when one root is pinned and only
  one physical slot is replaceable; both logical page entries and all
  residency counters remain unchanged;
- exhausting the one-page-per-frame upload limit leaves the second requested
  page missing and increments only the denial counter;
- exhausting a zero-eviction frame budget leaves the resident set and page
  table unchanged.

Retirement separately proves that the old mesh ID resolves as `retiring`, its
only mesh slot cannot be reallocated before GPU completion, and the same slot
can be reused only afterward with a different generation. The old ID then
resolves as stale.

## Regression boundary and gates

The complete quick lane passed in 57 seconds:

- 386 shared unit tests passed, one existing hot-reload test ignored;
- device negotiation passed;
- 59 golden tests passed, two hardware-specific tests ignored;
- four render-target tests and every visibility parity/runtime test passed;
- strict production correctness/suspicious/performance Clippy policy and
  formatting passed;
- FFI/schema parity, CI contracts, and file-size governance passed;
- 39 quality-governance tests, three visual fault-engine tests, and 29 cooker
  tests passed;
- baseline wasm, native `models3d`, and wasm `web,models3d` checks passed.

The no-default dependency graph still contains neither
`bloom-geometry-format` nor SHA-256; enabling `models3d` adds both explicitly.
Because no production owner calls or constructs the pool, the established
renderer has zero new passes, draws, buffers, bindings, allocations, shader
branches, or changed pixels. The full golden and live visibility GPU suites
confirm that boundary.

## Remaining acceptance work

No #131 acceptance checkbox changes at this checkpoint. The physical
allocation is exact, but driver-internal `Queue::write_buffer` staging overhead
has not yet been captured on integrated and discrete GPUs. GPU request
feedback, projected-error hierarchy traversal, frustum/cone/Hi-Z rejection,
indirect visibility submission, 10-million-triangle stress, motion/crack
qualification, and compatibility composition also remain open.
