# Issue #131 coarse-page prefix v1 evidence

This checkpoint qualifies deterministic coarse-first page placement at
revision `fe92915450dce3d45acc6a0192b4b9b0c3e3290c`. It builds on the atomic
hierarchy evidence and remains an offline-only artifact contract.

## Placement contract

- All coarse-root clusters are a prefix of the cluster table and page table.
- Roots and streamable clusters never share a page.
- Different hierarchy levels never share a page.
- Reordering remaps both sides of every parent/child group and fails if an
  atomic range would lose contiguity.
- The strict reader rejects mixed LOD/residency classes and any root page after
  the first streamable page.
- Cook and inspect reports expose root-page count and logical payload bytes.

A future loader can therefore upload a bounded page prefix without scanning
cluster records or accidentally pulling fine geometry into the always-resident
set.

## Canonical static-asset proof

Two independent release cooks of Damaged Helmet with
`--hierarchy-levels 8` are byte-identical:

- 100 level-3 roots occupy the first 8 pages;
- raw root-cluster payload: 468,570 bytes;
- logical root-page payload: 469,360 bytes;
- packing overhead: 790 bytes, far below one 65,536-byte page;
- fixed 64 KiB slot-allocation upper bound: 524,288 bytes;
- complete hierarchy payload remains 2,937,040 bytes;
- 48 total pages (one more page than mixed-level packing);
- largest page: 65,472 bytes under the 65,536-byte hard budget;
- complete artifact: 3,021,168 bytes;
- artifact SHA-256:
  `da0f68d731ca42a95a54e8ff157a62315a71dfa5c78d62b2bccd8ec59c0124f7`;
- payload SHA-256:
  `176aba193c2a0c4d8ac3910a3c4c9db4b661f5a0d8a234a5574317c8a51d6d30`.

Separating LOD classes adds one 64-byte page record and does not duplicate or
change geometry payload. It trades a negligible metadata increase for an
exact coarse residency boundary.

## Regression gates

- Seventeen release tests, formatting, and strict Clippy pass.
- A negative test corrupts one child LOD so a page mixes classes; the reader
  rejects it before exposing the archive.
- The default leaf cook remains byte-identical to its qualified artifact:
  `df089b23324fe8a8e00842a80b44894fd27b276c9124b8ff81dd77f8cf7b2cd2`.
- No engine/runtime file or dependency changes; current pixels and frame cost
  remain untouched.

This is not a measured runtime residency result. Allocation strategy, staging
overhead, eviction, missing-page fallback, and GPU traversal remain open.
