# Issue #136 runtime texture variant evidence

This checkpoint qualifies adapter-owned indexed texture selection and upload at
revision `9140c9672ce29af43840516646c0b768087850a4`; the core loader landed in
`cc557c0b966ba28b087c6f55b8362078f98784ad`. Qualification ran on an Apple
M1 Max (`arm64`) with macOS 26.5.

## Runtime contract

- `Renderer::cooked_texture_store_request` derives the package profile from
  the compiled platform and the features accepted by the actual `wgpu::Device`.
- Desktop devices with BC request their native same-quality profile and carry
  exactly one ordered `portable` fallback. Devices without accepted BC request
  `portable` directly. The shared plan also owns virtual-geometry requests.
- `CookedTextureStoreLoader` queues without blocking, then reads and validates
  `index.json`, the selected immutable chunk path, byte length, SHA-256, format,
  dimensions, mip count, DXGI header, single-layer 2D shape, and complete DDS
  surface layout on a bounded worker.
- An automatically selected portable profile may not contain BC. Indexed
  upload never silently decodes source data or regenerates mips; incompatibility
  is an actionable error.
- Runtime reports include requested and selected profiles, accepted BC state,
  runtime platform, selection kind, fallback rank, and a stable reason.
- Upload retains artifact semantics: normal/variance maps register as normal,
  sRGB as base color, and linear data as linear material data.

## Production end-to-end results

The release cooker rebuilt `embed-perry/bloomFull.png` (SHA-256
`b6727d85a0793a3434ceb2f012b4df5613a9be51da43afbb960077c281087aa6`)
into fresh temporary stores. Both stores passed `asset-index-inspect`, then the
release `cooked_texture_store_smoke` executable used the public renderer,
worker, texture-manager, and GPU-upload APIs.

Native exact case, with `macos/high` and `portable/high` installed:

- requested/selected: `macos/high` -> `macos/high`;
- selection: `exact`, reason `adapter-native-profile`;
- accepted BC: true;
- artifact: `bc7-rgba-unorm-srgb`, 880x600, 10 mips, 705,156 bytes;
- SHA-256:
  `bfdceade7a6be6c8cc59edffa9f7958c6bceeadba96e67ec13152722e3a83a61`;
- validated DDS uploaded successfully to the real Metal device.

Deliberate fallback case, with only `portable/high` installed:

- requested/selected: `macos/high` -> `portable/high`;
- selection: `fallback`, rank zero, reason
  `portable-fallback-after-native-miss`;
- artifact: `rgba8-unorm-srgb`, 880x600, 10 mips, 2,815,824 bytes;
- SHA-256:
  `df4aa2a4c036e669c83fd8cdccdc86b161cd6a952a8db5a09eda2a7defcbe8cb`;
- validated DDS uploaded successfully through the same public API.

The first native BC GPU control exposed and fixed a pre-existing compressed-
tail upload defect: WebGPU copies a virtual 2x2 or 1x1 BC mip as one physical
4x4 block. Passing the virtual extent caused an unaligned-copy validation
error. The uploader now uses the physical block extent, and the BC7 control,
portable control, and portable-after-native-miss control all pass on the GPU.

## Commands and qualification

The stores were cooked with `texture-store`, finalized with `asset-index`, and
validated with `asset-index-inspect`. Each was then loaded with:

```shell
cargo run --release --manifest-path native/shared/Cargo.toml \
  --example cooked_texture_store_smoke -- \
  "$store" qualification/bloom-full high
```

Qualification results:

- `bloom-cook`: 44 release tests passed; strict Clippy passed;
- native shared library: 480 release tests passed, 1 intentional hot-reload
  ignore; correctness/suspicious/performance Clippy passed;
- the focused runtime tests performed real GPU uploads for native BC7,
  portable RGBA8 without accepted BC, and portable fallback after native miss;
- no-default, models-only, and image-extras-only release builds passed;
- formatting and `git diff --check` passed;
- the file-line ratchet reports only the same five unrelated pre-existing
  oversized files; every new file remains below 2,000 lines.

This evidence qualifies issue #136's adapter/platform runtime selection and
deliberate fallback criterion for the native indexed runtime. The portable DDS
format is capability-neutral and remains loadable through the existing web
byte-upload path; browser package fetching/index transport and packed archives
remain separate documented integration work. The remaining issue #136
acceptance item is the full shipping-scene/source-free cook.
