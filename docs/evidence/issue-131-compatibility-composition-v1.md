# Issue #131 compatibility composition v1 evidence

This checkpoint qualifies mixed virtual/ordinary compatibility composition at
revisions `9a3027008daf384d9cac29be001b43574ca0ef39` and
`28939290cf711d31fc128f1b6eabffd494e0edc3` on Apple M1 Max Metal.

## Ownership and routing

Two strict source closures are decoded through the production archive reader.
The first contains an opaque virtual primitive and a cooker-routed
`alpha-blend` primitive. The second contains an opaque virtual primitive and a
cooker-routed `skinned` primitive. Runtime routing verifies each inspectable
reason and rejects incomplete or multiply owned partitions as before.

The compact compatibility cache now selects Bloom's established cached-skinned
draw whenever retained vertices contain skin weights. It consumes the staged
joint palette once for the compact model. Non-compact subset and arbitrary
full-transform submissions fail closed for weighted content: neither form can
preserve the world-space palette contract without also drawing virtual-owned
primitives.

Morph-target primitives remain a versioned, inspectable cooker exclusion. The
ordinary importer does not yet own morph animation, so #131 neither virtualizes
them nor silently claims to animate them.

## Real-GPU pixel oracle

The 128x128 native-full visibility test uses two overlapping pairs for each
qualified compatibility class:

- compatibility coverage behind virtual geometry matches the virtual-only
  pixel exactly, proving shared depth occlusion;
- front alpha-blended coverage changes the virtual pixel in the authored red
  blend direction;
- front skinned coverage changes the virtual pixel in the authored blue
  material direction;
- changing only the upper joint matrix changes more than 32 visible pixels,
  proving the routed skin is not frozen in bind pose;
- an enabled virtual renderer with an empty virtual batch remains pixel-exact
  to the ordinary frame.

The arbitrary-transform skinned call is also rejected before recording a draw.

## Qualification

- `cargo test --test visibility_buffer_shading_runtime -- --nocapture`: 1 pass;
- `cargo test --test golden_render cached_skinned_motion_sequence_bounds_animation_trails -- --nocapture`: 1 pass, zero trail frames, stable flicker 0.0360;
- `cargo test --lib`: 480 passes, zero failures, one existing ignored test;
- scoped Rust format and diff checks: pass.

The fixed cross-backend camera-motion corpus and complete ordinary glTF
invariance proof remain separate open #131 acceptance work.
