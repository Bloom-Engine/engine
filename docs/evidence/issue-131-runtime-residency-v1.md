# Issue #131 runtime archive and residency v1 evidence

This checkpoint qualifies the first CPU-side runtime milestone at revision
`c2a0bd2f78f8da7be4ee27a709a9bba900115f59`. It does not yet register virtual
meshes with the renderer or upload/traverse pages on the GPU.

## Shared fail-closed archive contract

The cooker and runtime now depend on the same `bloom-geometry-format` reader.
The prior cooker reader was extracted rather than copied, and the cooker still
runs it against every artifact before returning bytes. Runtime ownership adds:

- immutable archive bytes and decoded tables behind `Arc`;
- page byte ranges exposed only after complete validation;
- an indexed load that verifies complete-file length/SHA-256, format version,
  payload SHA-256, and source SHA-256 against the #136 artifact identity;
- useful and distinct format-versus-identity errors.

The established 29-test cooker suite remains green against the extracted
reader, including malformed magic/version/endian/ranges, payload/page hashes,
indices, packed vertex bits, hierarchy reciprocity, and page-class placement.
The new runtime test independently rejects corrupted magic and mismatched
indexed identity before exposing an asset.

## Fixed-budget residency and fallback proof

The end-to-end runtime fixture encodes a valid three-level hierarchy before
loading it through the production reader. It has eight clusters in five pages:
one 448-byte coarse-root page, two 224-byte intermediate pages, and two
448-byte leaf pages.

With a 1,120-byte cache budget:

- the 448-byte root prefix is pinned at construction;
- a missing leaf resolves deterministically to its level-2 root;
- after its 224-byte intermediate page is resident, it resolves to level 1;
- requesting the 448-byte leaf group evicts the least-recently-used 224-byte
  intermediate page and reaches exactly 1,120 resident bytes;
- the complete two-cluster leaf group then resolves exactly at level 0;
- telemetry records uploads, evictions, exact/fallback resolutions, pinned and
  resident bytes/pages.

A separate valid fixture splits one atomic leaf group across two 224-byte
pages. With 448 bytes pinned and only 224 streamable bytes available, the
448-byte group request fails and leaves every residency bit and counter
unchanged. A 447-byte total budget likewise rejects the 448-byte root prefix.

These are exact logical page-budget results. The #131 GPU-residency acceptance
criterion remains open until fixed GPU allocation, staging overhead, and
per-frame upload/eviction budgets are measured.

## Default-path neutrality

The module has no owner in `EngineState`, `Renderer`, `ModelManager`, the frame
graph, shaders, or FFI. Construction is explicit under `models3d`; the
no-default-features renderer benchmark dependency graph does not contain
`bloom-geometry-format` or SHA-256 code.

An exact detached-HEAD (`9cebeef`) versus working-tree comparison ran three
interleaved 1920×1080 Metal trials per side, each with 180 warmup and 240
measured uncapped frames. The complete `renderer_paths` JSON was identical in
all six runs (SHA-256
`6d71f4a20210d875593d16d2995909ec78b78314b9e2f5fc89d843dffe26389f`).
Median timings were all favorable to the working tree:

| Signal | Detached HEAD | Runtime milestone | Delta |
|---|---:|---:|---:|
| Wall mean | 6.8081 ms | 6.0525 ms | -11.10% |
| CPU mean | 1.8943 ms | 1.7283 ms | -8.76% |
| CPU p95 | 2.3776 ms | 2.1156 ms | -11.02% |
| GPU mean | 26.5489 ms | 18.8301 ms | -29.07% |
| GPU p95 | 30.7875 ms | 25.6282 ms | -16.76% |

The favorable magnitude is treated as run-to-run machine noise, not a claimed
speedup. Together with compile-time exclusion and identical renderer-path
telemetry, it rules out new default passes, allocations, branches, or frame
work.

## Gates and remaining work

- strict production Clippy and formatting pass for both the shared runtime and
  new format crate;
- contracts, FFI parity, file-size ratchet, and wasm checks pass;
- 382 shared unit tests pass (one existing ignore), including the three new
  runtime tests; device and golden/render-target suites pass;
- 29 cooker tests and 39 quality-governance tests pass;
- indexed `models3d` builds compile under native Metal and wasm32.

No additional #131 acceptance box is checked yet. Stable runtime mesh/page
IDs, asynchronous index lookup, real GPU page allocation/upload, projected-
error traversal, Hi-Z culling, indirect visibility submission, debug views,
10-million-triangle stress content, and discrete/integrated GPU qualification
remain open.

