# Issue #131 pinned-root page packing v1 evidence

This checkpoint corrects a residency-packing regression exposed by the
detailed Bistro virtual-geometry qualification. It changes only offline
geometry cooking and archive validation; the renderer adds no pass,
allocation, shader branch, or per-frame work.

## Fault and correction

Mesh-first streaming order keeps every instance's coarse roots in a compact
table span, but roots from one mesh can cross hierarchy levels. The page
encoder still classified every page by `(lod_level, coarse_root)`, so each
level transition started a new 64 KiB physical slot even though every root
page is pinned for the complete lifetime of the mesh.

On the detailed Bistro this placed 58,229,152 root payload bytes into 1,593
physical slots. A 128 MiB pool has 2,048 slots, leaving only 455 for detailed
streaming. The resulting fallback pressure produced a virtual-only blue
surface beside the menu sign and reduced direct ordinary/virtual parity to
0.87003 SSIM.

The encoder and strict reader now use the actual residency class:

- root and streamable clusters never share a page;
- streamable pages remain isolated by LOD level;
- pinned roots may share a page across LOD levels;
- roots remain a contiguous page prefix.

Geometry recipe v3 invalidates old cooked-cache keys. The governed wire
format remains version 2.

## Deterministic detailed-Bistro artifact

Two independent clean cooks produced byte-identical artifacts:

- artifact SHA-256:
  `593a4b6b31da0498873986ae9f9488779dee4f7db5734533e12a0fd25eca1c29`;
- payload SHA-256:
  `a7aade15d57f9aa10f1ffebda46ac4ec921bf91a2c26719002b97e72152afa02`;
- source-closure SHA-256:
  `5ac44350964992261184fee6db70f0962da1baf974f50d2bd8c367463c731289`;
- 261,627,824 bytes, 115,375 meshlets, and 3,828 pages;
- 905 pinned root pages carrying the unchanged 58,229,152 root payload
  bytes;
- 688 fewer root slots than the faulty cook, leaving 1,143 of 2,048 slots
  available for streamed detail before the pool fills;
- 44,032 fewer artifact bytes and 11,008 fewer runtime page-table bytes.

The detailed-Bistro regression gate now rejects more than 920 root pages and
requires at least 0.95 SSIM with at most 3.0 mean RGB error for the direct and
moving ordinary/virtual comparisons.

## Pixel, motion, and runtime qualification

The complete same-run ordinary/virtual qualification passed:

| Comparison | Mean RGB | SSIM | Missing geometry | Background leak |
|---|---:|---:|---:|---:|
| Ordinary vs virtual, direct | 0.930961 | 0.96734683 | 0.001736% | 0.000434% |
| Virtual direct vs returned | 0.163791 | 0.99208888 | 0% | 0% |
| Worst moving ordinary/virtual frame | 0.961510 | 0.96593132 | 0.000868% | 0.009983% |
| Worst matched-path return | 0.215067 | 0.99353906 | — | — |

At the direct virtual capture the fixed 128 MiB runtime pool had all 2,048
slots occupied, but fallback groups fell to 173 and pending groups to 168.
The selector emitted 14,078 clusters with zero missing-current-page,
selected, request, page-use, invalid-record, or depth-limit overflow. The
virtual-only blue surface is absent from the corrected capture.

## Scaling and regression gates

The release Metal stress retained the fixed 64 MiB stress pool and passed at
1, 10, and 100 placements of the 10-million-source-triangle corpus. At 100
placements it held 955 pages, selected 10,220 clusters, and reported:

- 5.265658 ms mean and 8.426207 ms p95 GPU frame time;
- 6.693620 ms mean wall frame time;
- 2.138291 ms hierarchy selection;
- 0.116362 ms draw emission;
- zero fallback groups, missing pages, IO failures, truncated requests,
  truncated page uses, or selector/request/page-use overflow.

Automated qualification:

- two independent detailed-Bistro cooks: byte-identical;
- focused mixed-level pinned-root test: pass;
- cooker: 35 passed; strict `-D warnings` Clippy: pass;
- geometry-format tests and documentation tests: pass;
- detailed-Bistro parity and camera-motion qualification: pass;
- full shared/models3d test command: pass (core library 471 passed, one
  ignored; GPU golden target 78 passed, two ignored; all integration targets
  passed);
- 10-million-triangle Metal stress and scaling gate: pass;
- formatting/diff whitespace: pass.

The historical coarse-page-prefix evidence remains valid for the root prefix,
root/streamable separation, and streamable LOD isolation. This checkpoint
supersedes only its stronger statement that roots of different LOD levels must
occupy different pages.
