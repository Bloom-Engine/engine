# Issue #136 deterministic asset index v1 evidence

This checkpoint qualifies a canonical validated loose-store index at revision
`2effe1e3fb05c4ec26943a55ee054ca6e66cf188`. It builds on the
content-addressed geometry manifests and gives future runtime work one bounded
lookup table without enabling runtime asset selection.

## Index contract

`asset-index <store>`:

- recursively scans only the `manifests/` tree;
- rejects symlinks, non-JSON files, non-UTF-8 paths, dot/parent components,
  and path/content logical-ID disagreement;
- revalidates every manifest recipe key, dependency record, artifact/file/
  payload hash, source closure, format, and strict `.bgeo` structure, then
  records the complete manifest hash;
- sorts entries by logical ID and emits no timestamp or output-root path;
- records logical ID/kind, recipe key, source hash, manifest path/hash, and
  immutable artifact path/hash/size/format;
- installs the index atomically and writes nothing when bytes are unchanged.

`asset-index-inspect` reconstructs the canonical index from the live store and
requires an exact byte match. A changed manifest therefore makes an old index
explicitly stale rather than silently routing to the prior artifact.

## Determinism, deduplication, and incremental proof

The canonical store contains two logical IDs for the same qualified Damaged
Helmet artifact:

- `quality/damaged-helmet`;
- `showcase/helmet-copy`.

The resulting index has:

- two sorted entries;
- one unique 1,446,496-byte chunk;
- 2,892,992 referenced bytes versus 1,446,496 unique bytes;
- 1,686 index bytes;
- SHA-256:
  `abab52afb3e9365afc36edc3fad322c993ff07e3ffb73a35cccd2410be21ff58`.

An independently populated clean store produces the exact same index bytes.
The first atomic index install measured 0.37 seconds with `/usr/bin/time -p`;
an unchanged fully revalidated build measured 0.03 seconds and wrote zero
indexes.

The automated scoped-update control changes one logical manifest's geometry
recipe. The other manifest remains byte-identical, inspection rejects the
prior index as stale, and rebuilding writes only the derived index. Corrupting
a referenced chunk prevents both index generation and index inspection.

## Gates and boundary

Twenty-five release tests, strict Clippy, formatting, and the file-line gate
pass. Tests cover deterministic ordering, dotted IDs, root escape rejection,
unexpected manifest-tree files, zero-write rebuilds, shared-chunk accounting,
scoped invalidation, stale-index rejection, self-inconsistent manifests, and
corrupt chunks.

No additional #136 acceptance box is checked. The index currently contains
geometry-only loose-store entries. Source path/license provenance,
non-geometry recipes, platform/quality variants, declarative multi-asset
builds, packed archives, garbage collection, runtime lookup, and asynchronous
streaming remain open.

Shipping runtime delta remains zero passes, draws, buffers, images, bindings,
shader branches, engine dependencies, pixels, and frame-time work.
