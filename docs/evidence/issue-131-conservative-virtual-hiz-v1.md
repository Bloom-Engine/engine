# Issue #131 conservative virtual Hi-Z v1 evidence

This checkpoint qualifies Bloom's opt-in previous-frame occlusion boundary for
virtual hierarchy traversal at revision
`e930a5e057ebe1059b8def5a2461f169a783b1a7`. It does not activate virtual
geometry by default or claim the remaining store-IO, Bistro motion,
stress-scale, or cross-backend milestones.

## Conservative visibility contract

An active virtual submission captures a private nine-level R32Float max-depth
pyramid after current-frame depth is complete. Its fixed 256x256 base and eight
downsamples own 349,524 texture bytes. A virtual-only frame linearizes only the
shared mip-0 source; it does not build the four screen-space downsample levels
used by SSAO and SSGI. Ordinary frames add neither the virtual resources nor a
frame-graph pass, and an enabled but idle virtual renderer records no capture.

The next traversal culls only a complete atomic hierarchy group whose every
in-frustum cluster is proven behind the farthest depth over its full projected
footprint. The query unions previous and current bounds, expands by two base
cells, and applies 2% relative plus 0.1 absolute linear-depth bias. Camera cuts,
resizes, skipped frames, new instances, near-plane/off-screen bounds, and more
than one base-cell of screen motion all fail open. Pending history becomes
eligible only after the command buffer is submitted.

Nine separate textures avoid the wgpu/Metal sampled-and-written mip hazard.
Captured instance identities use one compact sorted vector and binary search;
they do not allocate one tree node per instance. Asynchronously returned GPU
feedback publishes the latest visible, frustum-culled, occlusion-culled, and
occlusion-uncertain group counts without a render-loop wait.

## Real Metal oracles

A uniform max-depth source of 2.0 was captured with one stable virtual instance
at approximately depth 10. On the adjacent frame the production GPU traversal
selected zero clusters, reported exactly two occlusion-culled hierarchy groups,
and reported zero uncertain groups.

Moving the current view-projection footprint beyond the motion threshold made
both groups uncertain and visible, selecting all four fine clusters. A newly
appearing stable ID produced the same visible four-cluster result. Independent
history checks rejected camera cuts, a skipped frame, a render-extent change,
and explicit invalidation.

The production four-MRT renderer integration submitted two depth captures,
reported one captured instance and 349,524 texture bytes, and composed visible
virtual coverage without changing unrelated pixels. The disabled report kept
all virtual Hi-Z allocation and culling fields at zero.

## Qualification

The exact committed tree passed:

- 458 shared library tests, with one existing ignored test;
- the complete real-GPU golden corpus: 77 passed and two hardware-specific
  tests ignored;
- 38 focused virtual-geometry tests and the production four-MRT Metal test;
- native no-default and WebAssembly `web,models3d` checks;
- strict formatting and the repository quick/lint correctness and performance
  policy.

The file-size ratchet remains red only for the same three pre-existing files:
`renderer/shaders/post.rs`, `renderer/shaders/ssgi.rs`, and
`golden_render/temporal_history.rs`. Every file introduced or expanded by this
checkpoint is within its governed ceiling.

## Remaining acceptance work

No #131 acceptance checkbox changes at this checkpoint. Production acceptance
still needs asynchronous #136 store/index IO behind page feedback, approved
Bistro motion/parity qualification, a 10-million-source-triangle stress asset,
and integrated/discrete Metal, Vulkan, and Direct3D 12 timing and quality
evidence.
