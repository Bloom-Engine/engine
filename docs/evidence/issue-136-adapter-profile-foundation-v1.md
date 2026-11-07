# Issue #136 adapter-profile foundation evidence

This checkpoint qualifies the adapter-profile foundation at revision
`d5504aacf8f1a9a7cd6a6897b56c62308f984cb6`. Qualification ran on an Apple
M1 Max (`arm64`) with macOS 26.5. It does not claim that indexed logical
texture loading is complete.

## Qualified contract

- Texture recipe v3 cooks `portable` color/data profiles as capability-neutral
  RGBA8 DDS and native desktop profiles as BC7 DDS. Normal maps remain the
  quality-preserving RGBA8 direction/variance format on every profile.
- The renderer can construct a virtual-geometry package request from the
  accepted `wgpu::Device` features. A desktop device with accepted BC support
  requests its native platform and one ordered `portable` fallback; other
  devices request `portable` directly.
- Adapter-owned requests reject caller mutation, and every nontrivial runtime
  selection exposes a structured report with the requested and selected
  profiles, capability policy, fallback rank, and stable reason.
- The existing DDS uploader accepts both RGBA8 and BC7 package artifacts. BC7
  remains gated on the device's accepted `TEXTURE_COMPRESSION_BC` feature.

## Real package qualification

The rebuilt release cooker stored `embed-perry/bloomFull.png` (SHA-256
`b6727d85a0793a3434ceb2f012b4df5613a9be51da43afbb960077c281087aa6`)
twice under the same logical ID and quality:

- `portable/high`: `rgba8-unorm-srgb`, 2,815,824 bytes, artifact SHA-256
  `df4aa2a4c036e669c83fd8cdccdc86b161cd6a952a8db5a09eda2a7defcbe8cb`;
- `macos/high`: `bc7-rgba-unorm-srgb`, 705,156 bytes, artifact SHA-256
  `bfdceade7a6be6c8cc59edffa9f7958c6bceeadba96e67ec13152722e3a83a61`.

The variants had distinct recipe keys and immutable chunks. The generated v2
index contained two profiled entries and two unique chunks, totaling 3,520,980
bytes, and passed `asset-index-inspect`. Resolving `macos/high` selected the
BC7 entry exactly. Resolving missing `windows/high` with the single declared
`portable/high` fallback selected the RGBA8 entry at fallback rank zero.

The qualification used a temporary store and these production commands after
an explicit release build:

```shell
cargo build --release --manifest-path tools/bloom-cook/Cargo.toml

tools/bloom-cook/target/release/bloom-cook texture-store \
  qualification/bloom-full embed-perry/bloomFull.png "$store" \
  --platform portable --quality high
tools/bloom-cook/target/release/bloom-cook texture-store \
  qualification/bloom-full embed-perry/bloomFull.png "$store" \
  --platform macos --quality high
tools/bloom-cook/target/release/bloom-cook asset-index "$store"
tools/bloom-cook/target/release/bloom-cook asset-index-inspect "$store"
tools/bloom-cook/target/release/bloom-cook asset-resolve \
  qualification/bloom-full "$store" \
  --platform macos --quality high --fallback portable/high
tools/bloom-cook/target/release/bloom-cook asset-resolve \
  qualification/bloom-full "$store" \
  --platform windows --quality high --fallback portable/high
```

## Qualification

- `bloom-cook`: 44 release tests passed; strict Clippy and formatting passed;
- native shared library: 475 release tests passed, 1 intentional hot-reload
  ignore; correctness/suspicious/performance Clippy policy passed;
- `git diff --check`: passed;
- the file-line ratchet reports only the same five unrelated pre-existing
  oversized files; every changed source/documentation file remains below the
  2,000-line limit.

The automatic request behavior is covered by the native policy test, while the
package formats and resolution order are covered by the real store
qualification above. The remaining acceptance boundary is a generic indexed
texture runtime loader that consumes the selected logical-ID entry; until that
exists, issue #136's full runtime adapter/platform checkbox remains open.
