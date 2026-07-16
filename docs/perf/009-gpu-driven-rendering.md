# 009 — Indirect multi-draw for scene graph

**Effort:** ~1 week · **Expected gain:** Removes 68 CPU draw calls, enables GPU-side cull · **Status:** deferred

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

## Proposed approach

1. **One shared vertex buffer + one shared index buffer** for all scene
   geometry. On mesh upload, append vertices/indices into the shared buffers
   and record `(vertex_offset, index_offset, index_count)` per mesh.
2. **Per-draw descriptor buffer** (storage buffer): one struct per mesh
   containing `{ transform, material_idx, aabb, index_offset, index_count,
   vertex_offset }`. Updated from the scene graph in `prepare()`.
3. **GPU cull compute pass**: dispatch one thread per mesh. Each thread
   tests its mesh's AABB against the frustum (using the same
   `extract_frustum_planes` logic we use on the CPU today). Surviving draws
   append to an indirect-draw buffer via an atomic counter.
4. **Single `draw_indexed_indirect_count`** call in the scene render pass.
   GPU reads the indirect buffer, dispatches each surviving mesh.
5. **Material data** lives in a storage buffer indexed by `material_idx`,
   fetched per-draw in the vertex or fragment shader.

wgpu 24 supports `draw_indexed_indirect` and `draw_indexed_indirect_count`
via the `Features::INDIRECT_FIRST_INSTANCE` and `MULTI_DRAW_INDIRECT_COUNT`
feature flags. Check adapter support at device creation.

## References

- "GPU-Driven Rendering" (Haar & Aaltonen, SIGGRAPH 2015) — the
  ubisoft talk that kicked off the modern approach
- UE5 Nanite's "Cluster-based" variant — each cluster of triangles is a
  separate cull unit.
- NVIDIA GameWorks samples have a clean indirect-multi-draw demo.

## Acceptance

- Sponza main_hdr pass CPU time drops from ~700 µs to < 100 µs (measured via
  profiler's `main_hdr_pass` CPU phase).
- Frustum culling ratio (surviving draws / total meshes) logged per frame and
  reasonable (e.g. 30-70% culled on typical Sponza camera poses).
- Correctness: SSIM ≥ 0.99 vs baseline.
- Doesn't break on meshes that use different materials (material index is
  part of the descriptor).
- Graceful fallback when the adapter doesn't support multi-draw-indirect
  (write a TODO to handle — M1 Metal supports it).

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

## Deferred — reopen criteria

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
