# Cooked asset store

Bloom's #136 asset-database foundation stores virtual-geometry and cooked DDS
texture artifacts under deterministic recipe keys and immutable content hashes.
Optional platform and quality profiles provide an explicit variant contract
while preserving the byte-identical unprofiled v1 format. Native geometry
loading resolves the validated mixed-kind index off-thread and demand-pages
immutable geometry ranges; indexed texture loading, packed, network, and web
delivery remain future work.

## Build and inspect

```shell
cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  geometry-store world/sponza examples/sponza/assets/Sponza.glb out/assets \
  --platform macos --quality high \
  --hierarchy-levels 8 --vertex-format quantized32

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  texture-store world/sponza/albedo examples/sponza/assets/albedo.png out/assets \
  --platform macos --quality high

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  texture-store world/sponza/normal examples/sponza/assets/normal.png out/assets \
  --platform macos --quality high --normal

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

### Store/cache location

The final positional argument is the complete store and cache root. There is
no hidden user-global cache and no environment-variable default. For local
development, use an ignored repository-relative directory such as
`out/assets`; for CI, use a job cache or artifact directory such as
`.cache/bloom/assets`. The absolute root is deliberately absent from manifests
and indexes, so a validated store can move between machines without changing
its bytes.

Keep `manifests/`, `variants/`, and `chunks/` together while cooking and run
`asset-index-inspect` before packaging. A native geometry shipping package may
then retain only `index.json` and its referenced `chunks/`; source assets and
manifests are not read by `VirtualGeometryStoreLoader`. Keep the manifests in
build artifacts for provenance and verification. Old unreferenced chunks are
safe but are not yet garbage-collected automatically.

The store layout is:

```text
out/assets/
  manifests/world/sponza.json
  variants/macos/high/world/sponza.json
  chunks/sha256/<artifact-sha256>.bgeo
  chunks/sha256/<artifact-sha256>.dds
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

`bloom-asset-manifest-v1` records the common identity, dependency, recipe,
build-key, and immutable-artifact contract. Geometry manifests record:

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

Texture manifests use the `bloom-texture` recipe and record:

- the exact source-file SHA-256;
- whether the source is a normal map and whether it uses linear or sRGB color;
- a build-key SHA-256 over recipe version, source bytes, and those semantics;
- canonical DDS format, width, height, mip count, file hash, byte length, and
  immutable `.dds` path.

The existing direct `texture` and `texture-dir` commands share the same policy
as `texture-store`. Unprofiled and native-platform color/linear data use BC7;
the canonical `portable` profile uses RGBA8 so it is loadable without an
optional compression feature. Normal maps imply linear color and always use
RGBA8: RGB retains the exact authored direction at mip zero, vector-filtered
RGB stores a normalized direction at lower mips, and alpha stores accumulated
LEADR/Toksvig variance. A representative high-entropy Sponza normal map showed
visible direction damage under color-error-optimized BC7, so recipe v2 began
the normal-fidelity exception and recipe v3 makes portable color/data output
capability-neutral. Store commands reject unknown or duplicate texture
options instead of silently producing an ambiguous recipe.

### Manifest examples

These illustrative unprofiled manifests show the complete field shape. Hash
placeholders stand for canonical 64-character lowercase SHA-256 strings; real
files contain no comments or placeholders.

```json
{
  "artifact": {
    "bytes": 1446496,
    "format_version": 2,
    "path": "chunks/sha256/<artifact-sha256>.bgeo",
    "payload_sha256": "<payload-sha256>",
    "sha256": "<artifact-sha256>"
  },
  "build_key_sha256": "<build-key-sha256>",
  "dependencies": [
    { "kind": "source-closure", "sha256": "<source-sha256>" }
  ],
  "kind": "geometry",
  "logical_id": "quality/damaged-helmet",
  "recipe": { "name": "bloom-geometry", "version": 3 },
  "schema": "bloom-asset-manifest-v1",
  "settings": {
    "hierarchy_levels": 8,
    "max_triangles_per_meshlet": 124,
    "max_vertices_per_meshlet": 64,
    "page_budget_bytes": 65536,
    "vertex_format": "quantized32"
  },
  "source": { "sha256": "<source-sha256>" }
}
```

```json
{
  "artifact": {
    "bytes": 1398256,
    "format": "bc7-rgba-unorm-srgb",
    "height": 1024,
    "mip_levels": 11,
    "path": "chunks/sha256/<artifact-sha256>.dds",
    "sha256": "<artifact-sha256>",
    "width": 1024
  },
  "build_key_sha256": "<build-key-sha256>",
  "dependencies": [
    { "kind": "source-file", "sha256": "<source-sha256>" }
  ],
  "kind": "texture",
  "logical_id": "world/sponza/albedo",
  "recipe": { "name": "bloom-texture", "version": 3 },
  "schema": "bloom-asset-manifest-v1",
  "settings": { "color_space": "srgb", "normal_map": false },
  "source": { "sha256": "<source-sha256>" }
}
```

Profiled v2 manifests have the same kind-specific fields plus
`"profile": { "platform": "macos", "quality": "high" }`; the profile is
also part of the build key.

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
change output bytes must increment the relevant geometry or texture recipe
version even if the container remains readable.

## Asset benchmark commands

The benchmark commands emit machine-readable `bloom-asset-benchmark-v1` JSON.
Texture measurements report source/cooked disk bytes, complete mip-chain GPU
bytes, top-mip quality, source-decode versus DDS-parse time, CPU fallback
decode time, and offline encode time:

```shell
cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  texture-benchmark examples/intel-sponza/assets/textures/curtain_fabric_red_BaseColor.png \
  --iterations 15

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  texture-benchmark examples/intel-sponza/assets/textures/curtain_fabric_Normal.png \
  --normal --iterations 5
```

Geometry compares a source glTF import—including buffers and images—with a
full cooked-file read, structural validation, and payload hashes:

```shell
cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  geometry-load-benchmark examples/renderer-test/assets/DamagedHelmet.glb \
  /tmp/helmet.bgeo --iterations 25
```

Timing uses a warm OS cache and excludes GPU resource creation/upload. The DDS
"direct upload" measurement is container parsing only; it does not claim GPU
upload latency. Exact commands, thresholds, adapter/host context, and accepted
results live in the corresponding issue evidence rather than mutable prose.

## Incremental and corruption behavior

On a matching build key, `geometry-store` verifies all of the following before
reporting a cache hit:

- schema, kind, logical ID, recipe and settings;
- the build key recomputed from the manifest;
- canonical dependency records;
- canonical chunk path and declared length;
- complete chunk and payload hashes;
- strict `.bgeo` structure, version, source closure, and vertex format.

`texture-store` applies the same fail-closed cache policy to its source hash,
recipe/settings, canonical path, file hash/length, and parsed DDS format,
dimensions, array/depth shape, and mip count.

A valid hit writes zero chunks and zero manifests. A different source closure,
source texture, or setting produces a miss and a new immutable chunk;
unrelated logical manifests and chunks are untouched. Multiple logical IDs
with identical cooked bytes share one chunk. After an asset changes, the
installed package index deliberately fails stale inspection until one explicit
`asset-index` rebuild; that rebuild writes only `index.json`.

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

Each entry contains the logical ID and kind, recipe build key, source hash,
manifest path/hash, and kind-specific immutable artifact metadata. The index
contains no timestamps or output-root paths, so two clean stores with the same
logical manifests produce byte-identical indexes. Duplicate logical ID/profile
pairs cannot be represented by the canonical path mapping; path/content
disagreement fails validation.

An unchanged index writes nothing. `asset-index-inspect` rebuilds the expected
index in memory from the live manifest tree and requires the installed bytes
to match exactly. It therefore detects a stale index after one manifest
changes, as well as corrupt manifests or chunks, before future runtime lookup
could observe them.

The index build report distinguishes total referenced bytes from unique chunk
bytes. Several logical IDs may share one immutable chunk without hiding their
individual references; the same is true of several platform/quality variants.

## CI use

Treat the source-to-logical-ID command list as tracked build input. Invoke the
same commands for every clean or restored-cache job, then build and inspect the
index. Cooker cache hits are fully revalidated, so restoring a damaged cache
fails the job rather than silently recooking over it.

```shell
set -euo pipefail

store=.cache/bloom/assets

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  geometry-store quality/damaged-helmet \
  examples/renderer-test/assets/DamagedHelmet.glb "$store" \
  --platform portable --quality high \
  --hierarchy-levels 8 --vertex-format quantized32

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  texture-store quality/bloom-full embed-perry/bloomFull.png "$store" \
  --platform portable --quality high

cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  asset-index "$store"
cargo run --release --manifest-path tools/bloom-cook/Cargo.toml -- \
  asset-index-inspect "$store"
```

Use the cooker revision, target profile, and source-lock revision in the CI
cache key to control storage growth. That outer cache key is an optimization,
not a correctness boundary: every manifest still recomputes its recipe key and
validates its referenced chunk. Run the release `bloom-cook` tests in the tool
job when changing recipes or formats. Build the index only after all asset
commands finish; publish it together with every referenced chunk.

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

## Native runtime resolution

`VirtualGeometryStoreLoader` and `CookedTextureStoreLoader` are the source-free
native counterparts to `asset-resolve`. Each ignores the other declared asset
kind, owns one bounded worker, and exposes a non-blocking `request`/`poll`
interface. Explicit requests still supply the logical ID, requested
platform/quality, ordered fallbacks, and whether an unprofiled legacy entry is
allowed. Selection follows exactly the offline resolver contract: exact first,
caller-ordered fallbacks next, and unprofiled only when explicitly enabled.

Production renderer callers should instead use
`Renderer::virtual_geometry_store_request(logical_id, quality)` and
`Renderer::cooked_texture_store_request(logical_id, quality)`. Both use one
shared adapter-profile plan derived from the compiled runtime and the accepted
`wgpu::Device` features. A macOS/Windows/Linux device with accepted BC support
requests its native profile and carries exactly one same-quality `portable`
fallback. Other devices request `portable` directly. Neither silently lowers
quality or opts into unprofiled assets. Each automatic or fallback result logs
and exposes structured JSON containing the requested/selected profiles, BC
capability, runtime platform, selection kind, fallback rank, and a stable
reason such as `adapter-portable-profile` or
`portable-fallback-after-native-miss`.

The loader worker performs every potentially blocking startup operation:
reading and parsing `index.json`, checking its schema/count/duplicate
identities, resolving the variant, rejecting non-canonical or symlinked chunk
paths, and validating the immutable chunk's complete file/payload/source
identity. It retains only validated archive metadata and the coarse-root page
prefix. The update or render thread only enqueues and polls. Completed assets
use the existing `Arc<VirtualGeometryAsset>` registration path, so renderer
setup remains simple and existing direct-byte callers are unchanged.

The texture worker additionally validates the selected immutable file length
and SHA-256, canonical `.dds` path, declared format/dimensions/mip count, DXGI
container format, single-layer 2D shape, and complete surface layout. It
rejects BC inside an automatically selected `portable` profile or on a device
without accepted BC support. The update thread passes the completed result to
`TextureManager::load_resolved_cooked_texture`, which uses Bloom's established
DDS uploader. This indexed path deliberately has no source decode or mip-
regeneration fallback: a package/device mismatch is an actionable error.

After registration, GPU missing-page feedback queues exact file ranges on the
page worker. Each result is checked against its independent page SHA-256 before
the atomic group is eligible for upload. In-flight plus completed-but-not-yet-
uploaded payloads are bounded by both group count and byte budgets; budgeted
GPU upload/eviction remains unchanged. Corruption fails the requested group
closed while its pinned resident ancestor stays drawable.

This runtime intentionally does not rebuild the index from manifests. Shipping
stores may omit manifests and all source assets; `index.json` plus its immutable
`chunks/` references are the runtime contract. Installers and development
workflows should continue to run `asset-index-inspect` before packaging.

## Migration from direct source loading

Migration is deliberately incremental; source loading remains available for
development and fallback while each shipping path is qualified.

1. Add deterministic `geometry-store` and `texture-store` commands to the
   asset build, retaining the existing source files and logical naming.
2. Finish the build with `asset-index` and gate CI/release packaging with
   `asset-index-inspect`. Do not package a stale index.
3. For native virtual geometry, replace render-thread source glTF parsing with
   a bounded `VirtualGeometryStoreLoader::request`/`poll` flow. Use the
   renderer-owned request constructor for adapter selection, or build a fully
   explicit request when testing a particular fallback. Register the returned
   `Arc<VirtualGeometryAsset>` through the established pool path and retain its
   structured selection report in startup diagnostics.
4. For native textures, queue the renderer-owned request through
   `CookedTextureStoreLoader`, poll during an ordinary update, then pass the
   validated result to `TextureManager::load_resolved_cooked_texture`. Direct
   paths remain available during migration: Bloom's existing texture loader
   magic-sniffs DDS, and glTF image lookup can retry `foo.dds` when `foo.png`
   is absent.
5. Remove source glTF/images from a shipping target only after its cold-start,
   variant/fallback, and visual-quality tests pass on that target. Keep direct
   source loading enabled in development until the complete scene, material,
   skin/animation, and texture dependency set has a cooked runtime path.

If a runtime reports an unsupported manifest, index, geometry format, or
recipe-derived artifact, recook with the shipping executable's matching cooker
revision. Never rewrite version fields or weaken hash validation in place.

## Current boundary

This checkpoint remains loose-store-only. It does not yet add:

- material, animation, environment, or world recipes;
- source path/license provenance;
- dependency-graph invalidation across asset kinds (for example, a material
  depending on several textures);
- garbage collection, packed shipping archives, or network-backed stores.

The indexed native runtime loaders remain opt-in and loose-store-only. Web
package fetch/integration and packed archives remain later work. The automatic
profile constructors and validated texture upload path change no default
renderer path, buffers, shaders, passes, draws, pixels, or frame-time behavior.

The canonical variant, fallback, deduplication, and Bistro qualification is
recorded in `docs/evidence/issue-136-asset-variants-v2.{md,json}`.
