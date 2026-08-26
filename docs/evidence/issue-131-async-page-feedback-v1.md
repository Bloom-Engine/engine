# Issue #131 asynchronous page feedback v1 evidence

This checkpoint qualifies Bloom's bounded GPU-to-CPU missing-page feedback and
in-memory page upload boundary at revision
`660190b38de2845b6ac90093d42873f8dee0ae8a`. It does not yet claim store-backed
file IO, virtual occlusion, or complete #131 activation.

## Non-blocking bounded feedback

Virtual-geometry traversal copies its 48-byte counter block and a bounded
request prefix into two fixed `MAP_READ | COPY_DST` buffers. The production
default permits at most 4,096 16-byte requests per capture, so each buffer is
65,584 bytes and total feedback ownership is 131,168 bytes. The limit is
clamped to the selector's request capacity. The ordinary renderer allocates and
records none of this path unless virtual geometry is explicitly enabled.

Mapping begins only after queue submission. Frame start performs a non-blocking
device poll, consumes only completed mappings, and rejects an out-of-order
completion if newer camera feedback was already consumed. If both slots remain
busy, the frame skips feedback rather than waiting and continues drawing the
nearest resident ancestors.

Completed requests are canonicalized to one entry per mesh/source-cluster
group and kept in a fixed-capacity persistent queue. Newest camera feedback is
serviced first. Invalid or generation-stale mesh/page identities are removed;
budget-blocked groups stay pending while other groups can progress. Uploads use
the page pool's existing atomic-group contract and per-frame byte, page, and
eviction limits. Runtime telemetry exposes readback ownership, in-flight and
pending counts, truncation, stale completions, stalls, resolved groups, and
uploaded pages/bytes.

## Real Metal streaming oracle

The production selector traversed a hierarchy whose middle groups were
resident and whose two leaf groups were missing. It emitted exactly two page
requests while keeping the resident ancestors selected. After submission, the
feedback mapping was observably in flight rather than synchronously consumed.

The pool was limited to one upload page per frame. The first service frame made
one group resident and retained the second request after a budget stall. The
next frame made the second group resident. A final production GPU traversal
selected fine cluster-table records `[4, 5, 6, 7]`, emitted no page requests,
and reported no ancestor fallback. This proves that a busy or budget-limited
feedback path degrades to visible coarse geometry rather than holes.

## Qualification

The exact committed tree passed:

- 455 shared library tests, with one existing ignored test;
- the complete real-GPU golden corpus: 77 passed and two hardware-specific
  tests ignored;
- the focused asynchronous streaming Metal oracle and the production
  four-MRT virtual-renderer Metal integration oracle;
- native no-default and WebAssembly `web,models3d` checks;
- strict formatting and scoped correctness/suspicious/performance Clippy.

The general shared Clippy lane remains red only on eight pre-existing findings
in `input.rs`, `renderer/mod.rs`, `string_header.rs`, and `shadows.rs`; the
scoped run passes after allowing those exact baseline diagnostics. The
repository file-size ratchet remains red only for three pre-existing files:
`renderer/shaders/post.rs`, `renderer/shaders/ssgi.rs`, and
`golden_render/temporal_history.rs`. No file in this checkpoint exceeds its
governed ceiling.

## Remaining acceptance work

No #131 acceptance checkbox changes at this checkpoint. Production acceptance
still needs asynchronous #136 store/index IO behind this boundary,
conservative previous-frame virtual Hi-Z, approved Bistro motion/parity
qualification, a 10-million-source-triangle stress asset, and integrated and
discrete Metal, Vulkan, and Direct3D 12 timing and quality evidence.
