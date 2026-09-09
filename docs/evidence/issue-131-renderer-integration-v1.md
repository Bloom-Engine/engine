# Issue #131 renderer integration v1 evidence

This checkpoint qualifies the first production-renderer attachment of Bloom's
virtualized-geometry path at revision
`3a0e900267f6a0b8d9a1caa174de7e046e3a2fa7`. It adds explicit renderer
ownership, current-frame submission, shared depth/visibility routing, and
four-MRT PBR composition. The path remains opt-in and does not claim complete
#131 activation or Bistro parity.

## Ownership and public boundary

`Renderer::enable_virtual_geometry` constructs the fixed GPU page pool,
hierarchy selector, virtual draw emitter, visibility rasterizer, and PBR
consumer. The renderer exposes explicit disable, page-pool access, and
current-frame submission entry points. Enabling fails with a useful error unless
the renderer was created with `BLOOM_VISIBILITY_BUFFER=shade`.

An ordinary renderer leaves the owner as `None`, allocates no virtual resource,
records no virtual work, and does not alter pixels. A separate no-models build
retains the exact pre-checkpoint visibility-preparation implementation. After an
opt-in owner is created, submitting an empty virtual batch is pixel-exact with
the established visibility/forward result.

Capability and quality-capture JSON now report whether the owner is enabled,
whether the current frame requested and prepared virtual work, instance count,
submission mode, pool capacity and GPU bytes, resident pages, active meshes,
and bounds.

## Integrated frame order and composition

A current-frame submission runs hierarchy traversal and draw emission before
the depth pass. It forces creation of the existing packed `Rg32Uint` visibility
target even when the frame has no ordinary visibility-eligible draw. Virtual
rasterization then writes to that same target after ordinary and compatibility
depth, preserving their occlusion.

Ordinary and virtual identity namespaces are disjoint: the ordinary fullscreen
consumer discards virtual IDs and the virtual PBR consumer discards ordinary
IDs. Virtual PBR shading is recorded in the same HDR pass and writes the same
four production MRTs before forward compatibility geometry is composed. A
virtual shading bind group is invalidated and rebuilt when the shared target is
recreated or resized.

The renderer fails a submitted virtual batch closed if traversal, emission, or
visibility preparation fails. Virtual shading is additionally gated on a
successful virtual raster in the current frame, including frames skipped by a
diagnostic prepass, so stale virtual visibility cannot be composed.

## Real Metal integration oracle

The production visibility-shading runtime test now exercises this registered
path on Metal. It first proves that enabling the owner and submitting an empty
batch changes no pixel. It then registers a valid one-page, one-cluster cooked
triangle, binds a production material, submits one instance, and renders it
through traversal, emission, the shared visibility target, and four-MRT PBR
composition.

The virtual triangle changes more than 256 pixels, every changed pixel remains
inside its expected region, and unrelated ordinary and compatibility pixels
remain byte-exact. Runtime telemetry reports one requested and prepared
instance, one active mesh, and nonzero GPU ownership. Repeated frames create no
new visibility target or bind group after initialization.

## Qualification

The complete governed release lane passed:

- 448 shared library tests passed and one existing test was ignored;
- all 29 filtered virtual-geometry tests passed on Metal;
- one negotiated-device test, 77 golden tests, four render-target tests, two
  visibility parity tests, one visibility runtime test, and the production
  visibility-shading integration test passed;
- two hardware-specific golden tests and two doc tests remained ignored;
- strict formatting and correctness/suspicious/performance Clippy passed;
- the WebAssembly web build, 39 quality-governance tests, three visual
  fault-engine tests, 29 cooker tests, and the 20-example inventory passed;
- CI inventory and FFI/schema contract checks passed.

The repository-wide file-size ratchet remains red only for three pre-existing
files outside this slice: `renderer/shaders/post.rs`,
`renderer/shaders/ssgi.rs`, and `golden_render/temporal_history.rs`. No file
introduced or changed by this checkpoint exceeds its governed ceiling.

## Default-path performance boundary

The no-models/default path was compared against parent revision
`b92668f23f45b076660af56a92e4d72ab9c1599f` on an Apple M1 Max/Metal at
1280x720, with 180 warmup and 600 measured uncapped frames per run. Two paired
windows were run in opposite order.

The averaged render/submit results were 2.262720 ms for the parent and
2.255512 ms for the candidate mean (-0.32%), 2.468188 vs 2.488000 ms P50
(+0.80%), 2.949605 vs 3.052396 ms P95 (+3.48%, 0.103 ms), and 4.161605 vs
4.094459 ms P99 (-1.61%). The P95 direction reversed between the two orderings
and the remaining differences are within the observed host-noise envelope.
This establishes an effectively flat default-path checkpoint, not a performance
win claim.

## Remaining acceptance work

No #131 acceptance checkbox changes at this checkpoint. Production acceptance
still needs cooked compatibility-record routing with no holes, asynchronous
missing-page feedback and streaming, conservative previous-frame Hi-Z, an
approved Bistro motion/parity corpus, a 10-million-source-triangle stress asset,
and timing/quality qualification on integrated and discrete Metal, Vulkan, and
Direct3D 12 adapters.
