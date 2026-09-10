# Transmitted-shadow inverse camera matrix correction

The transmitted-shadow resolve uploaded the CPU inverse view-projection matrix
without converting its storage layout for WGSL matrix-vector multiplication.
Receiver positions were reconstructed in the wrong world coordinates, so the
pass missed the actual glass shadow and could tint unrelated boundary pixels.
The upload now transposes the inverse matrix, matching SSR, PT, and fog.

This defect surfaced in #156's macOS shared-test lane at `73e6945`: the profiler
regression passed, but the colored-shadow test failed with RGB losses
`(2345, 7628, 406)`. The original test included asynchronous GI, which could
switch from screen-space tracing to SDF tracing between its two short captures.
GI is now disabled in this direct-shadow fixture, and failed captures are
retained automatically. Its camera, light, material, frame counts, and assertions
remain unchanged.

The isolated fixture fails reproducibly on Radeon/Vulkan before the matrix
correction: RGB losses are `(19510, 37818, 966)`. Correcting the upload produces
a visible shadow on the floor beneath the glass, with 3,217 affected pixels
and losses `(99060, 70446, 471)`. The existing cyan-transmittance assertion passes.
A wrong-color control swaps the authored red and green attenuation values:
losses become `(68302, 95675, 656)` and the same assertion rejects it. The
temporary control is reverted. No approved image or assertion threshold changes.

An intermediate diagnostic also tested an oblique light and a constant resolve
strength. Those variants are retained as diagnostic evidence and are absent
from the final change. The production correction is one matrix transpose;
resource counts, pass structure, and shadow-map resolution are unchanged.

The local command is `cargo test --release --manifest-path native/shared/Cargo.toml
--test golden_render physical_transmission_casts_a_bounded_colored_directional_shadow
-- --nocapture`, with `WGPU_BACKEND=vulkan` and `BLOOM_REQUIRE_GPU=1`.

Raw logs, before/after PNGs, the wrong-color control, and source changes are
retained under `tools/quality/out/windows-engine-plan/profiler-integrity/colored-shadow-stage/`
and published with the [#156 evidence prerelease](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-profiler-windows-vulkan-20260910).
The initial profiler ZIP remains pinned to `73e6945`; the shadow follow-up
archive records its own source revision and hash manifest. Hosted validation
must be assessed against the follow-up PR head.
