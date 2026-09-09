# Issue #136 native runtime streaming v1 evidence

This checkpoint qualifies Bloom's native loose-store lookup and file-backed
virtual-geometry demand paging at revision
`1cc6f1a360c87144f2b7af063d9502281eb77dcc`. It connects the deterministic
#136 index to the existing #131 GPU missing-page feedback boundary without
adding filesystem work to the update or render thread.

## Runtime contract

`VirtualGeometryStoreLoader` owns one bounded native worker. Its non-blocking
`request`/`poll` API strictly validates `index.json`, applies the caller's exact
profile and ordered fallback policy, rejects symlinks and non-canonical chunk
paths, and validates the selected immutable artifact before registration.
Only archive metadata and the 50 coarse-root pages remain resident after the
temporary complete-file validation buffer is dropped.

GPU feedback for a non-root atomic group is sent to a separate bounded page
worker. The worker opens the artifact once for the group, reads only the exact
page ranges, and rechecks every independent page SHA-256. The renderer uploads
a group only after all payloads validate. It preserves the resident ancestor
on corruption, stale requests, I/O pressure, or GPU upload-budget pressure.

The default I/O envelope is 128 outstanding groups and 32 MiB across in-flight
and completed-but-not-uploaded payloads. Telemetry exposes outstanding/ready
groups, reserved bytes, requests, completions, failures, queue stalls, and
bytes read. Web builds compile without the filesystem worker; direct-byte
assets retain the established memory-backed path.

## Deterministic 10M-source-triangle proof

The qualified store contains the exact 582,052,704-byte artifact from the
#131 10M workload under `chunks/sha256/<artifact-sha256>.bgeo`. Its artifact
SHA-256 is
`45b6aa47ce817589911d33f3f4b32387847cf28a9baf6c94523ae1f81db198d8`;
the 984-byte v2 index SHA-256 is
`8ad290a59a2d5f2c6a652127f9045b2bf564b2d43560b2ccce851b89acffe97f`.
`asset-index-inspect` accepted the canonical store.

The real Metal runtime warmed for 180 moving-camera frames and measured 120
more at 640x360. It retained the fixed 64 MiB physical pool and settled at 955
resident pages. File-backed streaming issued and completed 370 atomic reads,
read 76,226,352 bytes rather than retaining the 582 MB artifact, uploaded 905
fine pages, and reported zero I/O failures, pending groups, missing-current
pages, fallbacks, overflows, or invalid records. Reserved I/O bytes returned to
zero at the end of the gate.

| Measurement | Result | Enforced maximum |
|---|---:|---:|
| Caller-side request/poll maximum | 0.1239 ms | 50 ms |
| Wall frame mean | 6.7390 ms | 16.6667 ms |
| GPU frame mean | 4.4761 ms | 8.0 ms |
| GPU frame p95 | 7.8572 ms | 12.0 ms |
| Hierarchy selection GPU mean | 1.6054 ms | 3.0 ms |
| Draw emission GPU mean | 0.1115 ms | 0.5 ms |

The asynchronous initial validation completed in 8.439 seconds. That work is
fully off-thread and intentionally hashes and structurally validates the
complete artifact before returning an asset. Reducing this cold-load cost
without weakening the trusted-index boundary remains a separate optimization.

## Automated qualification

- focused virtual-geometry library tests: 34 passed;
- complete shared library: 465 passed, one expected ignored;
- complete real-GPU golden renderer corpus: 77 passed, two expected ignored;
- file-backed 10M stress gate: passed;
- strict shared correctness/suspicious/performance Clippy: passed;
- WebAssembly `web,models3d` check: passed;
- formatting and diff whitespace: passed.

The file-size ratchet remains red only for the same three unrelated existing
files: `renderer/shaders/post.rs`, `renderer/shaders/ssgi.rs`, and
`tests/golden_render/temporal_history.rs`. This checkpoint adds no overage.

## Remaining boundary

This is native loose-file geometry streaming. Packed shipping archives,
network-backed stores, web fetch/decompression, non-geometry recipes, garbage
collection, and source-license provenance remain open #136 work. The next #131
qualification slice is discrete-GPU and cross-backend timing and quality.
