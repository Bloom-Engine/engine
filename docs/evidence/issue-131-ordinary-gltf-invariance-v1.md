# Issue #131 ordinary glTF invariance v1 evidence

This checkpoint qualifies the ordinary-model no-op boundary at revision
`405d383bcb800ab0b1e819f70cfd1b4d339141de` on Apple M1 Max Metal.

## Production asset path

The real-GPU visibility runtime test loads the checked-in canonical
`examples/renderer-test/assets/DamagedHelmet.glb` through
`load_model_with_textures_from_source_path`, retains its imported material and
mesh transform data, and submits it through Bloom's established cached-model
path. The ordinary glTF contributes 328 visible pixels relative to the same
procedural control frame, so a missing or off-screen model cannot make the
comparison pass vacuously.

## Exact no-op oracle

Two 128x128 frames use the same camera, clear color, ambient light, procedural
content, and ordinary glTF draw. The second frame only enables the virtual
renderer and submits an empty virtual-geometry batch. The result is byte-exact:

- changed pixels: 0;
- maximum channel delta: 0;
- changed-pixel bounds: empty.

The established procedural-pixel, virtual-depth, translucent-compatibility,
and skinned-compatibility assertions continue in the same real-GPU test after
the ordinary glTF is added.

## Isolation boundary

TAA, SSAO, SSR, SSGI, bloom, motion blur, sharpening, and auto exposure are
disabled, and render scale is fixed at 1.0 for this oracle. That isolates
virtual renderer registration and mixed composition from independent temporal
or post-processing state; it does not claim those effects are generally
pixel-invariant.

## Qualification

- `cargo test --test visibility_buffer_shading_runtime -- --nocapture`: 1 pass,
  328 visible glTF pixels, zero no-op pixel differences;
- `cargo test --lib`: 480 passes, zero failures, one existing ignored test;
- scoped Rust format and diff checks: pass.

The fixed cross-backend camera-motion corpus remains the open #131 acceptance
work.
