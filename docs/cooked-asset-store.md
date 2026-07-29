# Cooked asset store

Bloom's first #136 asset-database checkpoint stores virtual-geometry artifacts
under deterministic recipe keys and immutable content hashes. It is an
offline-only building block: the shipping runtime does not select or load this
store yet.

## Build and inspect

```shell
cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  geometry-store world/sponza examples/sponza/assets/Sponza.glb out/assets \
  --hierarchy-levels 8 --vertex-format quantized32

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  asset-inspect world/sponza out/assets

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  asset-index out/assets

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  asset-index-inspect out/assets
```

Logical IDs are relative slash-separated ASCII identifiers. Empty components,
`.`/`..`, absolute paths, backslashes, and non-ASCII characters are rejected.
Dots within a component are preserved; `chair.v2` and `chair.v3` cannot map to
the same manifest.

The store layout is:

```text
out/assets/
  manifests/world/sponza.json
  chunks/sha256/<artifact-sha256>.bgeo
  index.json
```

Manifests are installed only after their chunk is flushed and strictly
validated. Chunks are immutable: if a file already exists at a content-hash
path but its bytes or hash differ, the command fails instead of overwriting
it. Manifest replacement uses the same rollback-safe atomic writer as direct
geometry cooking.

## Recipe and manifest contract

`bloom-asset-manifest-v1` records:

- logical ID and asset kind;
- source-closure SHA-256, covering the glTF/GLB and every resolved buffer;
- geometry recipe name/version;
- max meshlet vertices/triangles, page budget, hierarchy levels, and vertex
  format;
- a build-key SHA-256 over the recipe version, source closure, and every
  setting above;
- canonical source dependency records;
- relative immutable chunk path, file/payload hashes, byte length, and
  `.bgeo` format version.

The manifest intentionally omits local source and store paths so an identical
logical ID, source closure, and recipe produces byte-identical manifests in a
different clean output directory. The build report includes the local input
and manifest paths for diagnostics. Source path/license provenance, platform
and quality variants, and a package index remain later #136 work.

Recipe version changes are explicit. A future cooker behavior change that can
change output bytes must increment the geometry recipe version even if the
container remains readable.

## Incremental and corruption behavior

On a matching build key, `geometry-store` verifies all of the following before
reporting a cache hit:

- schema, kind, logical ID, recipe and settings;
- the build key recomputed from the manifest;
- canonical dependency records;
- canonical chunk path and declared length;
- complete chunk and payload hashes;
- strict `.bgeo` structure, version, source closure, and vertex format.

A valid hit writes zero chunks and zero manifests. A different source closure
or setting produces a miss and a new immutable chunk; unrelated logical
manifests are untouched. Multiple logical IDs with identical cooked bytes
share one chunk.

Malformed manifests and corrupt referenced chunks fail closed. They are not
silently treated as cache misses, because doing so would hide damage to an
installed database. `asset-inspect` runs the same self-consistency and chunk
validation without requiring source assets.

## Canonical store index

`asset-index` recursively discovers the manifest tree, rejects symlinks and
unexpected files, derives each logical ID from its canonical manifest path,
and runs the complete manifest/chunk inspection above. It then writes
`bloom-asset-index-v1` with entries sorted by logical ID.

Each entry contains the logical ID and kind, recipe build key, source-closure
hash, manifest path/hash, and immutable artifact path/hash/size/format. The
index contains no timestamps or output-root paths, so two clean stores with
the same logical manifests produce byte-identical indexes. Duplicate logical
IDs cannot be represented by the canonical path mapping; path/content
disagreement fails validation.

An unchanged index writes nothing. `asset-index-inspect` rebuilds the expected
index in memory from the live manifest tree and requires the installed bytes
to match exactly. It therefore detects a stale index after one manifest
changes, as well as corrupt manifests or chunks, before future runtime lookup
could observe them.

The index build report distinguishes total referenced bytes from unique chunk
bytes. Several logical IDs may share one immutable chunk without hiding their
individual references.

## Current boundary

This checkpoint is geometry-only and loose-store-only. It does not yet add:

- texture, material, animation, environment, or world recipes;
- source path/license provenance;
- platform/quality variant selection;
- dependency-graph invalidation beyond one geometry source closure;
- garbage collection or packed shipping archives;
- runtime lookup, asynchronous IO, residency, or fallback.

It changes no production renderer path, buffers, shaders, passes, draws,
pixels, or frame-time behavior.
