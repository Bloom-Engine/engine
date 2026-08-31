# Issue #136 cooked asset variants v2 evidence

This checkpoint qualifies explicit platform/quality variants and ordered
fallback at revision `8bfa90580dbb60a439ad8cdff91d53d441fb7fa1`.
Unprofiled stores retain byte-identical v1 manifests and indexes; a store with
profiles uses `bloom-asset-manifest-v2` and `bloom-asset-index-v2`.

## Contract

- `--platform` and `--quality` are paired, bounded canonical identifiers.
- The profile participates in the recipe build key but not the immutable
  artifact hash, so compatible profiles cannot collide semantically and still
  deduplicate identical bytes.
- Index entries sort by logical ID, platform, then quality.
- Resolution tries the exact request first, then only caller-authored
  `--fallback PLATFORM/QUALITY` values in order. Legacy fallback requires the
  separate `--allow-unprofiled` opt-in.
- Index build, inspection, and resolution revalidate every manifest, source
  closure, recipe key, chunk path/hash, and `.bgeo` payload before selection.

## Bistro store qualification

The asset is pinned to Bistro revision
`7c9f9f9ac0915024ccf3dddbccd8bfc643a42607`. The qualification uses the 16
largest finite camera-visible unique meshes from the authored camera. The
resulting quantized, eight-level virtual-geometry artifact is 13,015,712 bytes
with SHA-256
`4d6120ce64101f53d54f5d3707d84d15fdb3aed3a7ab6c7ee0482ab96e42dd92`.

`macos/high` and `portable/high` produce distinct build keys but share that one
chunk. The two-entry index references 26,031,424 bytes while storing 13,015,712
unique bytes. Its 1,929 canonical bytes hash to
`1212607e840cd023a9d872870ed0fbfb98881d43171abb1e7436062315db9af6`.
Rebuilding a clean store in the opposite insertion order produces an exact
byte match. A repeated `macos/high` cook is a zero-write cache hit. Exact macOS
resolution succeeds; `windows/ultra` fails over to `portable/high` only when
that fallback is explicitly supplied.

The 64- and 96-mesh Bistro derived sets are deliberately not claimed as
cooker-qualified: both fail closed on an authored non-finite position (vertex
8 of the selected source mesh). Runtime asset sanitation/provenance remains a
separate #136 follow-up; validation was not weakened to make the fixture pass.

## Gates and runtime boundary

Twenty-nine release cooker tests, strict Clippy, formatting, and the repository
file-line gate pass. The tests cover v1 byte compatibility, deterministic v2
ordering, distinct profile keys, shared artifact deduplication, zero-write
hits, exact/fallback/legacy selection, path/profile disagreement, missing
variants, and corrupt stores.

This is still an offline contract. Shipping runtime delta remains zero passes,
draws, buffers, textures, bindings, allocations, shader branches, pixels, and
frame-time work.

