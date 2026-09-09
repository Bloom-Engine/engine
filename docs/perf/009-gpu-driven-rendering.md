# 009 — Indirect multi-draw for scene graph

**Effort:** ~1 week · **Expected gain:** Removes CPU draw loops, enables GPU-side cull · **Status:** landed (2026-07-23)

## Problem

The scene graph issues one CPU draw call per mesh (`scene.render()` loops
over every node and calls `pass.draw_indexed(...)`). On Sponza that's 68
draws per frame per pass — shadow pass does it 3× (once per cascade),
depth prepass (ticket 005) will do it once, main_hdr once. ~340 CPU draw
calls per frame.

The CPU wins landed already (uniform pool, frustum cull) cut most of the
per-draw overhead, but we still have 340 `set_bind_group` calls. GPU-driven
rendering collapses this to **one `draw_indirect_count` call** — the GPU does
the culling and dispatches its own draws.

## Landed approach

1. **One shared vertex buffer + one shared index buffer** for all scene
   geometry. On mesh upload, append vertices/indices into the shared buffers
   and record `(vertex_offset, index_offset, index_count)` per mesh.
2. **Per-draw descriptor buffer** (storage buffer): one struct per mesh
   containing `{ transform, material_idx, aabb, index_offset, index_count,
   vertex_offset }`. Updated from the scene graph in `prepare()`.
3. **GPU cull compute pass**: dispatch one thread per mesh. Each thread
   tests its mesh's AABB against the frustum (using the same
   `extract_frustum_planes` logic we use on the CPU today). Commands retain
   deterministic scene order; culled slots get `instance_count = 0`.
4. **Single indexed multi-draw-indirect call** in each depth/main render
   pass. Metal uses fixed-count `multi_draw_indexed_indirect`; adapters with
   `MULTI_DRAW_INDIRECT_COUNT` use the count variant.
5. **Material data** lives in a storage buffer indexed by `material_idx`,
   fetched per-draw in the vertex or fragment shader.

wgpu 29 supports indexed multi-draw and indirect-count submission
via the `Features::INDIRECT_FIRST_INSTANCE` and `MULTI_DRAW_INDIRECT_COUNT`
feature flags. Check adapter support at device creation.

The fast path requires Tier-A global material indirection and at least 32
eligible draws. Smaller scenes keep the lower-overhead CPU loop. Skinned,
active-LOD, and order-sensitive retained alpha scenes also stay on the
compatibility path. The `BLOOM_GPU_DRIVEN=0` environment override remains an
explicit qualification oracle.

## Qualification result

Apple M1 Max / Metal, fixed 1,280×720 quality captures:

- 10,240-draw stress: one indirect call, 10,180 visible and 60 culled;
- stress CPU frame mean: 75.79 ms compatibility → 15.83 ms GPU-driven;
- stress GPU frame mean: 16.31 ms → 10.41 ms;
- stress `main_hdr_pass` CPU: 2.324 ms → 0.182 ms;
- Sponza `main_hdr_pass` CPU: 0.014 ms, below the 0.100 ms target;
- final, HDR, depth, and three shadow captures for the stress case are
  byte-identical to the compatibility path;
- full seven-case corpus minimum SSIM is 0.999908 across 42
  final/intermediate comparisons; every shadow capture is exact;
- PBR spheres exercise 25 distinct materials; the 10k stress exercises 12
  shared material records with per-draw tint.
- `cargo check` passes for shared, macOS, Linux, and the actual iOS, tvOS,
  visionOS, and WebAssembly targets. Android and Windows cross-checks reach
  their platform C dependencies but require the NDK/MSVC SDKs, which are not
  installed on the qualification host.

Run artifacts are written under `tools/quality/out/issue-28-*` and are ignored
by git.

## References

- "GPU-Driven Rendering" (Haar & Aaltonen, SIGGRAPH 2015) — the
  ubisoft talk that kicked off the modern approach
- UE5 Nanite's "Cluster-based" variant — each cluster of triangles is a
  separate cull unit.
- NVIDIA GameWorks samples have a clean indirect-multi-draw demo.

## Acceptance

- [x] Sponza `main_hdr_pass` CPU time is < 100 µs.
- [x] Submitted, compatibility, visible, culled, and cull-ratio telemetry is
  emitted in `renderer_paths.gpu_driven`.
- [x] Correctness SSIM is ≥ 0.99 (observed corpus minimum: 0.999908).
- [x] Heterogeneous material IDs are part of each draw descriptor.
- [x] Unsupported adapters and non-profitable/unsafe draw classes fall back
  without changing the compatibility renderer.

## Notes for the implementer

- This is a separate win from depth prepass (ticket 005); compose well —
  prepass and main pass can share the same indirect draw buffer.
- Skinned meshes need their joint matrices fetched per-draw — extend the
  descriptor or keep skinning in a separate pass.
- Biggest risk: materials need to be bound globally (bindless textures) or
  the fragment shader still needs per-material bind group switches. On
  wgpu/Metal, bindless is limited — may need a texture-array trick.

## Files likely to change

- `native/shared/src/renderer/mod.rs` (the old single `renderer.rs` was
  split into the `renderer/` module) — shared VB/IB, descriptor buffer, GPU
  cull compute shader, new render pass using `draw_indexed_indirect_count`.
- `native/shared/src/scene.rs` — reworking of per-node GPU resources.

## Historical deferred rationale

Pure CPU-side optimization: removes ~340 CPU draw calls/frame on
Sponza. But the perf README's own rule of thumb applies — **Sponza is
GPU-bound, not CPU-bound**. The prior CPU-side wins (uniform pool,
frustum cull, matrix-inverse cache from commit 95da6af) already cut
render-total CPU to ~4 ms against a 16.7 ms vsync budget. Shaving
another ~600 µs of CPU via draw-call collapsing **won't move FPS on
the current benchmark** — we'd be optimizing a resource we already have
in surplus.

Reopen when:

- **A CPU-bound scene arrives** — 10 000+ mesh count, many small
  static props, or CPU-expensive per-frame state updates that push
  `render_total` CPU past the vsync budget.
- **Ticket 008 (visibility buffer) starts.** 008's shading pass needs
  a shared vertex/index buffer + per-mesh descriptor buffer — exactly
  what this ticket builds. If 008 reopens, this ticket is a hard
  prerequisite and should land first.
- **Bindless texture support lands in wgpu.** The current "one
  `set_bind_group` per draw" pattern is partly about per-material
  texture binds. With bindless, indirect multi-draw becomes a
  straightforward win without the material-binding workarounds the
  ticket's "Notes for the implementer" describes.

Estimated effort when reopening: ~1 week for the baseline
`draw_indexed_indirect_count` path with GPU frustum cull. Material
indirection still requires either bindless (not widely supported in
wgpu 29) or a texture-array trick — that's where the ticket's risk
sits, and why it's scoped at "week" not "days."
