# Cooked asset store

Bloom's #136 asset-database foundation stores virtual-geometry artifacts under
deterministic recipe keys and immutable content hashes. Optional platform and
quality profiles provide an explicit variant contract while preserving the
byte-identical unprofiled v1 format. This remains an offline/runtime-contract
building block: the shipping renderer does not load the store yet.

## Build and inspect

```shell
cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  geometry-store world/sponza examples/sponza/assets/Sponza.glb out/assets \
  --platform macos --quality high \
  --hierarchy-levels 8 --vertex-format quantized32

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  asset-inspect world/sponza out/assets --platform macos --quality high

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  asset-index out/assets

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  asset-index-inspect out/assets

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  asset-resolve world/sponza out/assets --platform windows --quality ultra \
  --fallback portable/high
```

Logical IDs are relative slash-separated ASCII identifiers. Empty components,
`.`/`..`, absolute paths, backslashes, and non-ASCII characters are rejected.
Dots within a component are preserved; `chair.v2` and `chair.v3` cannot map to
the same manifest.

The store layout is:

```text
out/assets/
  manifests/world/sponza.json
  variants/macos/high/world/sponza.json
  chunks/sha256/<artifact-sha256>.bgeo
  index.json
```

Manifests are installed only after their chunk is flushed and strictly
validated. Chunks are immutable: if a file already exists at a content-hash
path but its bytes or hash differ, the command fails instead of overwriting
it. Manifest replacement uses the same rollback-safe atomic writer as direct
geometry cooking.

The `manifests/` tree is the byte-compatible unprofiled v1 layout. Profiled
manifests live under `variants/<platform>/<quality>/`; a store may contain
both during migration. Profile identifiers are bounded lowercase ASCII tokens
and cannot escape the store.

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

`bloom-asset-manifest-v2` adds a canonical `{ platform, quality }` profile.
The profile is part of the recipe build key even when two profiles currently
produce identical geometry bytes. Those artifacts still deduplicate to one
immutable chunk, while future platform-specific settings cannot collide in
the cache. `--platform` and `--quality` must be supplied together; omitting
both retains v1 behavior and bytes.

The manifest intentionally omits local source and store paths so an identical
logical ID, source closure, and recipe produces byte-identical manifests in a
different clean output directory. The build report includes the local input
and manifest paths for diagnostics. Source path/license provenance remains
later #136 work.

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

`asset-index` recursively discovers the legacy and profiled manifest trees,
rejects symlinks and unexpected files, derives each logical ID/profile from
its canonical path, and runs the complete manifest/chunk inspection above.
An unprofiled-only store retains byte-identical `bloom-asset-index-v1` output.
A store with any profiled entry writes `bloom-asset-index-v2`, sorted by
logical ID and then platform/quality, with each profiled entry carrying its
canonical profile.

Each entry contains the logical ID and kind, recipe build key, source-closure
hash, manifest path/hash, and immutable artifact path/hash/size/format. The
index contains no timestamps or output-root paths, so two clean stores with
the same logical manifests produce byte-identical indexes. Duplicate logical
ID/profile pairs cannot be represented by the canonical path mapping;
path/content disagreement fails validation.

An unchanged index writes nothing. `asset-index-inspect` rebuilds the expected
index in memory from the live manifest tree and requires the installed bytes
to match exactly. It therefore detects a stale index after one manifest
changes, as well as corrupt manifests or chunks, before future runtime lookup
could observe them.

The index build report distinguishes total referenced bytes from unique chunk
bytes. Several logical IDs may share one immutable chunk without hiding their
individual references; the same is true of several platform/quality variants.

## Explicit variant resolution

`asset-resolve` validates the installed index against every live manifest and
chunk before selecting an entry. The requested platform/quality pair is tried
first. Each `--fallback PLATFORM/QUALITY` is then considered in command-line
order. An unprofiled v1 entry is considered only with
`--allow-unprofiled`.

The machine-readable result labels the selection `exact`, `fallback`, or
`unprofiled-fallback`, includes the selected profile and fallback rank, and
returns the validated manifest/artifact identity. If nothing allowed exists,
the command fails and lists the available profiles. There is deliberately no
implicit cross-platform, lower-quality, or legacy fallback.

## Current boundary

This checkpoint is geometry-only and loose-store-only. It does not yet add:

- texture, material, animation, environment, or world recipes;
- source path/license provenance;
- dependency-graph invalidation beyond one geometry source closure;
- garbage collection or packed shipping archives;
- production runtime index loading, asynchronous IO, or residency.

It changes no production renderer path, buffers, shaders, passes, draws,
pixels, or frame-time behavior.
