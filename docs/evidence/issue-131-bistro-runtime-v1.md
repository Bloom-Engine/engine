# Issue #131 detailed Bistro runtime v1 evidence

This checkpoint qualifies Bloom's opt-in virtual-geometry path on the detailed
Bistro at revision `5c26c6ab92e913949f15b68f8709c0504bc39f29`. It compares
ordinary and virtual ownership at the same native test resolution, exercises
streaming during camera motion, and returns to the exact starting camera. The
gate specifically catches the missing roads, façade holes, color drift, and
camera-dependent disappearance found during qualification.

## Corrected contracts

Four independent faults were exposed by the real scene:

1. Multi-level hierarchy siblings retained their individual lower child
   ranges after being merged into a new atomic group. Runtime traversal uses
   one common child range per atomic group, so later refinement followed the
   first sibling and dropped the others. The cooker now partitions only across
   contiguous compatible replacement ranges, coalesces the complete lower
   union, writes it to every sibling, and makes every grandchild point back at
   the complete group.
2. Per-cluster bounding spheres could classify all members of an atomic group
   outside different frustum planes even though the group crossed the view.
   CPU and GPU traversal now test the exact transformed AABB and cull the group
   only when every sibling is outside the same plane. A real GPU/CPU parity
   oracle covers the straddling case.
3. Virtual visibility discarded back faces for a non-double-sided glTF
   material, while Bloom's established opaque scene path rasterizes both
   faces. Reversed and mirrored imported surfaces therefore vanished only
   after virtual residency arrived. Opaque virtual visibility now follows the
   same two-face raster contract; alpha-masked content remains compatibility-
   owned and the face bit remains available to shading.
4. The runtime importer bakes `baseColorFactor`, including its legacy specular-
   glossiness conversion, into vertex color. The cooker emitted raw `COLOR_0`
   or white instead. It now performs the identical conversion and multiply,
   with a focused GLB import regression test.

## Workload and artifact

The source is `BistroReference.gltf` from the detailed Bistro benchmark. It has
1,176 placements and 2,803,489 eligible source triangles. Runtime routing sends
1,074 placements through virtual geometry and preserves 102 compatibility
placements.

The governed version-2 `quantized32` artifact is 261,628,080 bytes. It contains
115,375 clusters in 3,832 pages, reaches LOD level 7, and pins 909 root pages
(58,229,152 bytes). The payload SHA-256 is
`561efc237a8e210c2788a23f03e12bcc506370b7e24d9a51de2501d69ae7e179`;
the complete source-closure SHA-256 is
`5ac44350964992261184fee6db70f0962da1baf974f50d2bd8c367463c731289`.

The Metal runtime uses a fixed 128 MiB physical page pool. At the qualified
starting camera it had 1,957 resident pages, no pending groups, no fallback or
missing-current-page events, and no selected/request/invalid/depth-limit
overflow. The selector reported 10,689 selected clusters, 595 refined groups,
69 cone-culled clusters, 2,027 frustum-culled groups, and 523 conservatively
occlusion-culled groups.

## Pixel and motion gate

The environment-gated integration test renders ordinary and virtual children
in separate processes at 640x360. The virtual child warms streaming for 180
frames, captures the starting view, moves through 30 camera steps, returns
through 30 steps, settles for 30 frames, and captures the same view again.

| Comparison | Mean RGB | SSIM | Clearly-lit missing geometry |
|---|---:|---:|---:|
| Ordinary vs virtual after warmup | 7.135473 | 0.81424849 | 0.001736% |
| Virtual direct vs returned camera | 0.000314 | 0.99999140 | 0% |

The enforced thresholds are mean RGB at most 8, ordinary/virtual SSIM at least
0.80, clearly-lit missing geometry at most 0.5%, and camera-return SSIM at
least 0.985. Intermediate captures demonstrate that the gate is sensitive to
each diagnosed class: the original hierarchy measured 10.7369 mean RGB and
0.76937 SSIM; hierarchy and face fixes without cooked material factors still
measured 9.3034 and 0.79258. Both fail the final gate.

## Automated qualification

- detailed Bistro 180-frame parity/motion gate: pass;
- shared library: 459 passed, one existing ignored;
- focused virtual-geometry corpus: 39 passed on the real Metal device;
- GPU golden renderer corpus: 77 passed, two hardware/manual tests ignored;
- cooker: 31 passed, strict `-D warnings` Clippy passed;
- shared strict correctness/suspicious/performance Clippy: pass;
- wasm `web` check and native no-default-feature check: pass;
- Rust formatting and diff whitespace: pass.

The changed virtual-geometry files remain within the 2,000-line ratchet. The
repository-wide ratchet still reports three unrelated pre-existing overages in
`renderer/shaders/post.rs`, `renderer/shaders/ssgi.rs`, and
`tests/golden_render/temporal_history.rs`; this checkpoint adds none.

## Remaining #131 work

This closes the detailed-Bistro runtime parity/motion qualification slice. It
does not claim the 10-million-source-triangle residency stress, asynchronous
#136 store/index IO, or integrated/discrete/cross-backend timing. Those remain
the next independent gates before virtual geometry can become a default path.
