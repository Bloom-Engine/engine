# Issue #136 source-free shipping scene evidence

This checkpoint qualifies the last issue #136 acceptance item at revision
`1420e5a307dfe9b36654730fdbeef44097215d76`. Qualification ran on an Apple
M1 Max (`arm64`) with macOS 26.5. The machine-readable measurements and full
hashes are in `issue-136-shipping-scenes-v1.json` beside this document.

## Shipping contract

- `bloom-scene-format` is the shared, versioned `.bscene` definition used by
  cooker and runtime. Its fixed header records magic, format version, payload
  length, and payload SHA-256; decoding revalidates the complete canonical
  archive before exposing offsets or model data.
- The payload mirrors Bloom's runtime contract rather than retaining glTF
  JSON: unique primitives, placements/transforms, vertices and UV1, complete
  current material metadata, ordered texture dependencies, bounds, source
  closure, optional skeleton/clips, and deterministic sanitation diagnostics.
- `scene-store` uses the established staged glTF importer offline, supports
  GLB and loose external buffers/images (including Bistro DDS siblings), cooks
  every texture dependency, and emits one mixed-kind canonical index.
- The native scene boundary selects an exact or explicitly authorized profile,
  validates index identity and budgets, rejects non-canonical/symlink chunk
  paths, validates the complete `.bscene` file and payload hashes, and rebuilds
  the runtime model only after receiving the ordered cooked-texture handles.
- Texture recipe v4 serializes the renderer's exact alpha-coverage mip chain
  for masked materials. Normal maps retain the prior vector/variance policy.

The packaged-input control temporarily removed each store's entire `variants/`
manifest tree before running the native smoke. Every smoke passed using only
`index.json` and referenced `chunks/sha256/*`. The runtime path has no source
path parameter and reported `source_gltf_reads: 0`; source glTF, external
buffers/images, and cooker manifests were unavailable through the store.

The smoke validates every indexed DDS dependency through the existing cooked
texture loader, then reconstructs the model with nonzero placeholder handles.
The earlier runtime-texture checkpoint separately qualified real BC7 and RGBA8
GPU upload through the same validated texture artifacts; this scene checkpoint
does not claim that loading all 16.27 GB of Bistro textures into GPU memory is
a useful test.

## Four-scene results

| Scene | Primitives / placements | Textures | Scene bytes | Texture artifact bytes | Skin/animation | Result |
|---|---:|---:|---:|---:|---:|---|
| Damaged Helmet | 1 / 1 | 5 | 1,583,687 | 111,848,840 | 0 / 0 | pass |
| Fox | 1 / 1 | 1 | 242,225 | 5,592,552 | 24 joints / 3 clips | pass |
| Sponza | 103 / 103 | 69 | 21,660,807 | 380,293,768 | 0 / 0 | pass |
| Bistro | 551 / 2,909 | 680 | 188,317,493 | 16,266,801,464 | 0 joints / 7 clip records | pass |

All four used `portable/high`, passed `asset-index-inspect`, selected the exact
profile, validated every scene and texture hash, rebuilt the expected runtime
primitive/placement counts, and passed again without build manifests.

The source scenes contain non-finite shading attribute lanes. The cooker did
not hide them: manifests and archives record 14,556 repairs for Helmet, 1,728
for Fox, 192,496 for Sponza, and 1,738,262 for full Bistro. All source positions
were finite, so no triangle, primitive, or placement was dropped in any scene.
The shipping archives contain only finite runtime attributes. Invalid positions
would drop only affected triangles; invalid transforms or an empty result fail
closed.

## Commands and controls

Each source was cooked and indexed with the equivalent of:

```shell
bloom-cook scene-store scenes/<id> <source.glb-or-gltf> "$store" \
  --platform portable --quality high
bloom-cook asset-index "$store"
bloom-cook asset-index-inspect "$store"

mv "$store/variants" "$store/variants.build-only"
cooked_scene_store_smoke "$store" scenes/<id> portable high
mv "$store/variants.build-only" "$store/variants"
```

The deterministic regression performs two independent clean cooks of a GLB,
requires identical scene/payload hashes, verifies the second cook is a zero-
write hit, builds and loads the mixed index through the native API, then
damages the immutable chunk and requires hash rejection. Format tests also
damage the payload and version, with incompatible versions returning an
actionable recook error. The alpha-coverage regression serializes the custom
mip chain to DDS and verifies the final mip retains authored 75% coverage and
does not bleed transparent-border color.

Qualification results:

- shared scene format: 2/2 release tests and strict Clippy passed;
- `bloom-cook`: 46/46 release tests and strict Clippy passed;
- native shared library: 480 passed, 1 intentional hot-reload ignore;
- native correctness/suspicious/performance Clippy passed;
- no-default, models-only, image-extras-only, and combined models/image release
  builds passed;
- formatting and `git diff --check` passed;
- the file-line ratchet reports only the same five unrelated pre-existing
  oversized files; `models.rs` remains at the 2,000-line ceiling and every new
  file is below it.

This qualifies Sponza, full Bistro, Damaged Helmet, and the skinned Fox model
without runtime source glTF parsing in shipping mode. It closes the final
acceptance item for issue #136; packed archives, garbage collection, broader
provenance, environment cooking, and browser package transport remain useful
follow-up work rather than unfulfilled criteria in this checkpoint.
