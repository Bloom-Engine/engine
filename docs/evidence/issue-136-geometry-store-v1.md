# Issue #136 content-addressed geometry store v1 evidence

This checkpoint qualifies Bloom's first deterministic asset-database slice at
revision `289ba31ab6ff3853b474063af4dcc6df5ac7515f`. It stores the
already-qualified #131 geometry output behind a canonical recipe key and
logical manifest; it does not add a shipping runtime path.

## Contract

`geometry-store <logical-id> <source> <store>`:

- hashes an explicit geometry recipe version, the complete source closure,
  meshlet limits, page budget, hierarchy levels, and vertex format;
- writes immutable `.bgeo` chunks under their complete SHA-256;
- installs the logical manifest only after the chunk is flushed and strictly
  validated;
- preserves identical chunks across different logical IDs;
- reports a cache hit only after recomputing the manifest key and validating
  schema, settings, dependencies, chunk/file/payload hashes, source closure,
  format version, and complete `.bgeo` structure;
- writes zero files on a valid cache hit;
- fails closed on a malformed manifest or corrupt referenced chunk.

`asset-inspect` performs the same manifest/chunk validation without source
assets. Logical IDs cannot escape the manifest directory, and dotted IDs do
not collide.

## Determinism and incremental proof

Damaged Helmet was built with
`--hierarchy-levels 8 --vertex-format quantized32`:

- source closure:
  `3de5114323cba08aaef85757a90ed9685f1597b5ea0e7d7913f9fa45eeacaae7`;
- recipe key:
  `a5d53b58ff3275de76058602a8b2e699fdecdcaecacb417360583a6633e69273`;
- artifact:
  `6c8f924e8dad74a5acd8e9acfb21795b153559cedb880862eb0a7f23eee5bc62`;
- payload:
  `3450ff9074ddee3adda5d1373ddbe17d9a492c7130f8cae41a965dc4a2486169`;
- artifact size: 1,446,496 bytes;
- logical manifest:
  `efe37c5a986935ea3dfda9c08a8b8b53632b1c8152a08399a3577ae0f405fef6`,
  1,041 bytes.

Two clean output stores produced byte-identical manifests and chunks. A second
logical ID referencing the same source/settings wrote one manifest and zero
chunks; the store still contained exactly one chunk.

Warm release timings from `/usr/bin/time -p`:

- first build: 0.88 seconds;
- verified cache hit: 0.05 seconds;
- observed speedup: 17.6x;
- cache-hit writes: zero chunks, zero manifests.

These are offline wall-clock measurements, not renderer frame timings. The
cache hit still imports and hashes the source closure so external buffer
changes cannot be missed, but skips meshlet/hierarchy construction, encoding,
and all writes.

## Automated gates and boundary

Twenty-three release tests, strict Clippy, formatting, and the file-line gate
pass. The store test covers canonical IDs, zero-write hits, recipe-setting
invalidation, chunk deduplication, unrelated-manifest isolation,
self-inconsistent manifest rejection, and corrupt-chunk rejection.

No #136 acceptance box is checked yet. This is one geometry recipe in a loose
store, not the requested complete multi-asset database. Source path/license
provenance, textures/materials/animations/environments, platform and quality
variants, package/index dependency invalidation, shipping archives, runtime
lookup, and async streaming remain open.

Shipping runtime delta is zero render passes, draws, buffers, images,
bindings, shader branches, engine dependencies, pixels, and frame-time work.
