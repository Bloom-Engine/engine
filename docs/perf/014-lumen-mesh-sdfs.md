# 014 — Lumen per-mesh SDFs + global SDF clipmap + WSRC

**Effort:** 3-4 weeks · **Expected gain:** SW parity with HW for off-screen GI · **Status:** deprioritized

Phase 3 of the [Lumen roadmap](lumen-roadmap.md). Depends on 013.

**Priority note:** With HW ray-tracing live on macOS / iOS / tvOS / Windows /
Linux after 007b + 013, the SW SDF path is only critical for **Android and
web**. Deprioritise behind 016 (importance sampling) and only land if
Android / web traction demands it.

## Problem

007a's SW path Hi-Z-marches against screen depth and therefore cannot see
off-screen geometry. Lumen's SW equivalent is a two-tier SDF system:

1. **Per-mesh distance fields (MDFs)** — one small SDF volume baked per mesh.
2. **Global SDF clipmap** — sparse 3D clipmap around the camera that composites
   the relevant MDFs into a merged SDF for long-range traces.

Probe rays sphere-trace the SDF; hits shade from the Surface Cache (ticket
013). Rays that travel > 2 m without hitting fall through to a separate
low-resolution **World Space Radiance Cache (WSRC)** — clipmap probes at
32×32 octahedral resolution holding pre-integrated distant lighting.

## Approach

### Per-mesh SDF bake

At model load, rasterize each mesh into a 3D texture (typical size 32³ to
64³ per mesh) using GPU jump-flood or a CPU `mesh-to-sdf`-equivalent crate.
Cache to disk keyed by mesh content hash so re-loads skip the bake.

Pipeline choice: GPU jump-flood on a voxelized surface. Starts from seed
voxels on the mesh surface and propagates distance outward in log(max-dim)
passes. ~1 ms per 32³ mesh on Apple Silicon.

### Global SDF clipmap

Sparse 3D clipmap of merged SDFs around the camera:

- 4 cascades (2 m, 8 m, 32 m, 128 m half-widths).
- Each cascade: a sparse 64³ brick grid; bricks allocated only where meshes
  exist.
- Per-frame: for meshes within each cascade, sample their MDF into the brick
  grid with `min()` merge. Update only dirty bricks (static scene = nearly
  zero per frame).

### Sphere-trace shader variant

`SSGI_PROBE_TRACE_SDF_WGSL` — third trace variant alongside `_SW` (Hi-Z) and
`_HW` (ray-query). Selected via adapter + config:

```
if hw_rt_enabled              -> HW
else if sdf_enabled           -> SDF
else                          -> Hi-Z screen
```

Inner loop: sphere-trace the global SDF clipmap, stepping by the SDF value
clamped to cascade texel size. On hit, sample the Surface Cache (ticket 013)
for radiance.

### World Space Radiance Cache (WSRC)

- 3D clipmap of probes (separate from the global SDF clipmap) around the
  camera.
- Each probe: 32×32 octahedral atlas (higher directional resolution than
  SSRC because distant lighting is low-frequency in position but high in
  direction).
- Persistent across frames; refreshed at cascade-specific rates.
- Sampled when a screen-probe ray travels > 2 m from the screen without
  hitting — the ray reads the nearest WSRC probe's octahedral atlas along
  its direction.
- If even WSRC misses → sample analytic sky.

## Files likely to change

- `native/shared/src/models.rs` — SDF bake hook at load; cache to disk.
- `native/shared/src/renderer/mod.rs` — clipmap manager, brick grid
  textures, WSRC probe textures, SDF merge pass, WSRC refresh pass.
- `native/shared/src/renderer/shaders.rs` — `SDF_JUMP_FLOOD_WGSL`,
  `SDF_CLIPMAP_MERGE_WGSL`, `WSRC_REFRESH_WGSL`, `SSGI_PROBE_TRACE_SDF_WGSL`.
- New `native/shared/src/sdf_cache.rs` — disk cache for baked MDFs.

## Acceptance

- `./examples/intel-sponza/main --quality 3 --ssgi 1 --fps-only 300` with
  HW forcibly disabled (`BLOOM_FORCE_SW_GI=1`): off-screen bleed visible and
  within 20% of the HW path visual quality.
- FPS at same config: ≥ 40 fps (SDF is cheaper than HW on some hardware).
- Mesh SDF bake time: < 5 seconds total for Sponza's 68 meshes. Cached bakes
  load in < 100 ms.
- VRAM overhead for global SDF clipmap + WSRC: < 128 MB.
- WebGPU build compiles with this path enabled (this is the whole reason the
  ticket exists for web).

## Notes

- Do not attempt a sparse brick allocator from scratch — start dense per
  cascade (`64³ × 4 cascades × 2 bytes = 2 MB`) and only sparsify if VRAM
  pressure demands it.
- SDF traces are slower than HW rays but faster than Hi-Z over long distances
  because the step size is SDF-bounded, not pyramid-bounded. Expect SDF and
  HW to be within ~1.5× of each other on desktop GPUs.
- Mesh-to-SDF crates to evaluate: `mesh-to-sdf`, `sdfu`, hand-rolled GPU
  jump-flood. Benchmark before committing.
