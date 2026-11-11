# wgpu-core 29.0.3, patched for Bloom

Upstream crates.io `wgpu-core` 29.0.3 with one change in
`src/command/ray_tracing.rs` (search for "Bloom patch"):

- A BLAS created with `AccelerationStructureUpdateMode::PreferUpdate` and
  `AccelerationStructureFlags::ALLOW_UPDATE`, once a submitted command buffer
  has built it, is refitted in place on later builds (hal `Update` mode with
  the BLAS as its own source). Upstream 29 always does a full rebuild.
- The scratch offset uses the larger of the build and update scratch sizes.

Used by the Apple platform crates through `[patch.crates-io]`. Bloom rebuilds
its skinned-mesh BLASes every frame; see `DYN_BLAS_REFIT` in
`native/shared/src/renderer/mod.rs`. Drop this copy when upstream implements
BLAS updates.
