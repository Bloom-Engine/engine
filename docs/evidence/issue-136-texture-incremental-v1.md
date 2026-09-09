# Issue #136 incremental texture-store evidence

This checkpoint qualifies source-texture isolation at revision
`d24f17ecd60ba958c8c32a081577577185405461`. It adds a deterministic
`bloom-texture` recipe to the existing loose content-addressed store without
changing the default renderer or source-asset development path.

## Recipe and validation contract

- `texture-store <logical-id> <source> <store>` emits an immutable BC7 DDS
  chunk plus a canonical v1 or profiled v2 logical manifest.
- The recipe key covers the exact source-file SHA-256, recipe version,
  normal-map semantic, linear/sRGB semantic, and optional platform/quality
  profile.
- The artifact records and validates BC7 format, width, height, mip count,
  byte length, file hash, 2D depth, and array-layer count.
- Cache hits revalidate the complete manifest and DDS and write zero chunks
  and zero manifests. Corrupt metadata or chunks fail closed.
- Direct `texture`, `texture-dir`, and `texture-store` use one encoder; unknown
  and duplicate texture options are rejected.
- The canonical store index accepts geometry and texture entries. The native
  virtual-geometry loader validates common entry identity, accepts the
  declared texture kind, and ignores it while retaining strict geometry
  parsing.

## Incremental mixed-package qualification

The release regression creates two deterministic 8x8 source textures and one
minimal geometry source, then installs three logical assets into one package.
The first index contains three entries, three unique chunks, and the sorted
kinds `geometry`, `texture`, `texture`.

An unchanged cook of the first texture is a verified cache hit with zero chunk
and manifest writes. The test then edits only that source texture and proves:

- exactly one new texture chunk and one texture manifest are written;
- its build key and artifact hash both change;
- the other texture manifest and immutable DDS chunk remain byte-identical;
- the geometry manifest and immutable `.bgeo` chunk remain byte-identical;
- the installed `index.json` remains byte-identical and strict inspection
  rejects it as stale;
- one explicit `asset-index` invocation performs exactly one index write;
- the unrelated texture and geometry bytes remain identical after rebuilding;
- corrupting the new DDS chunk produces a hash-mismatch failure.

A separate profiled regression proves that linear-data and normal-map recipes
have distinct semantic build keys even when their identical linear BC7 output
deduplicates to one chunk. A native regression loads geometry successfully
from a synthetic mixed geometry/texture index.

## Qualification

- `bloom-cook` release tests: 37 passed, 0 failed.
- `bloom-cook` strict correctness/suspicious/performance Clippy: passed.
- native shared release library tests: 472 passed, 0 failed, 1 intentional
  hot-reload ignore.
- native shared strict correctness/suspicious/performance Clippy: passed.
- native and cooker formatting plus `git diff --check`: passed.

The repository-wide file-line ratchet remains red on five unrelated existing
renderer/virtual-geometry files. None is touched by this checkpoint; the new
files are 147 and 722 lines, below the 2,000-line ceiling.

This qualifies only the source-texture incremental rebuild acceptance item.
Material-to-texture dependency graphs, platform-specific texture encodings,
indexed runtime texture resolution, packed archives, garbage collection,
provenance, and texture memory/quality benchmarking remain open.
