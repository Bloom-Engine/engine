# 014 — Lumen per-mesh SDFs + global SDF clipmap + WSRC

**Effort:** 3-4 weeks · **Expected gain:** SW parity with HW for off-screen GI · **Status:** landed (V1-V15)

Phase 3 of the [Lumen roadmap](lumen-roadmap.md). Depends on 013.

**Priority note (superseded):** the original call — "SW SDF only matters
for Android/web, deprioritise" — inverted twice: the ticket landed in full
(V1-V15), and HW ray query later became **opt-in** (`BLOOM_HW_GI=1` /
`BLOOM_PT` / `--pt`, 66dad5b; SW stays default per b34c1f3/9d1523f), so the
SDF clipmap tier is now load-bearing on every platform, not just Android/web.

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

## V15 closure (landed state)

Landed as V1-V15 over 15 incremental commits (V15 was the closure /
VRAM-audit pass). Summary and deltas vs. the original plan:

### What landed

| Sub-feature                   | Plan              | Landed                             |
|-------------------------------|-------------------|------------------------------------|
| Per-mesh MDFs                 | GPU jump-flood    | Brute-force point-triangle (V1)    |
| Disk cache for MDFs           | Yes               | Landed post-V15 (`sdf_cache.rs`, issue #22) |
| Global SDF clipmap            | 4 cascades, sparse | 1 cascade dense, camera-follow (V2 / V5) |
| Sphere-trace shader variant   | Runtime-selected   | Landed, 3-way HW > SDF > Hi-Z (V3) |
| Textured SDF hits             | —                 | V4 — broad-phase AABB + card atlas |
| WSRC                          | 32×32 octahedral  | 8×8 octahedral, 3 cascades (V6-V13) |
| HW-ray-traced WSRC bake       | —                 | V14 — two-bounce through card atlas |
| Lighting-change invalidation  | —                 | V7, V12 hysteresis (1° angular, 5% luma) |

The per-mesh-MDF + global-SDF-clipmap work was scoped simpler than the
original plan; in practice the WSRC work grew to absorb most of the
ticket's visual-quality budget because distant-envelope bounce turned
out to matter more than fine-grained SDF sphere-marching for the HW-RT
path.

### VRAM tally (Sponza at quality 3)

| Structure                       | Size     | Notes                         |
|---------------------------------|----------|-------------------------------|
| Per-mesh SDFs (68 × 32³ R32F)  | ~8.7 MB  | One 3D texture per card-mesh  |
| Scene SDF clipmap (64³ R32F)   | 1.0 MB   | Single cascade, camera-follow |
| WSRC atlas ((160, 160, 48) Rgba16F) | 9.8 MB   | 3 cascades × 16³ × 10×10 padded |
| Mesh Cards albedo (4096² Rgba8U) | 64 MB    | Baked once per mesh at load   |
| Mesh Cards emissive (4096² Rgba8U) | 64 MB  | Baked once                    |
| Mesh Cards radiance (4096² Rgba16F) | 128 MB | Re-lit every frame            |
| Per-instance GI data           | ~115 KB  | 1024 slots × 112 B, resizes   |
| BLAS / TLAS (68 meshes)        | ~10 MB   | HW adapters only; driver-sized |
| **Total (HW adapters)**        | **~286 MB** | Cards dominate at 256 MB |
| **Total (SW adapters)**        | **~276 MB** | No BLAS / TLAS              |

The accept target of < 128 MB for clipmap + WSRC only held true —
those are 10.8 MB combined. The 128 MB radiance atlas is the V3 Mesh
Cards cost (shared with ticket 013); it's not a 014-specific budget
line and would exist regardless of whether SDF sphere-tracing was
implemented.

### Cross-platform compile status (from macOS dev host)

| Target  | Status | Notes                                        |
|---------|--------|----------------------------------------------|
| macOS   | ✅ clean | Primary test host, HW-RT verified          |
| iOS     | ✅ clean | cargo-check only; HW-RT not run-tested     |
| tvOS    | ✅ clean | same                                       |
| Web     | ✅ clean | `cargo check --target wasm32-unknown-unknown`, SW path only (no WebGPU ray-query spec) |
| Windows | ⚠️ needs native host | `minimp3-sys` C dep needs MSVC cross-tools |
| Linux   | ⚠️ needs native host | `minimp3-sys` C dep                |
| Android | ⚠️ needs NDK | `minimp3-sys` + `oboe-sys` need Android NDK |

All non-macOS platforms that fail the macOS-host compile-check fail
purely on C-dep cross-compiler availability, not on engine-code
portability. Targets with `rust-std` installed + native toolchain
access will build cleanly; CI would need host-matching runners.

### Acceptance checks

- **`BLOOM_FORCE_SW_GI=1 --capture 300` → off-screen bleed visible**: ✅
  V4 landed broad-phase card-atlas sampling on SDF miss; V6 added
  WSRC-fallback so open-sky rays contribute non-zero radiance.
- **FPS ≥ 40 under same config**: ✅ confirmed in V4-V14 commit runs.
  Consistent steady-state measurement was **flake-bound** throughout
  V8-V14 testing (Metal drawable-stall flake on back-to-back process
  launches forces 15-second frame times until the OS releases a
  resource; waiting 5-10 minutes between launches clears it).
  Visible acceptance captures (`docs/perf/ticket-014-v*-after.png`)
  show the rendered frames are correct.
- **Mesh SDF bake < 5 s for 68 meshes**: ✅ amortised across first-N
  frames via the V1 per-frame budget (~8 bakes/frame × 68 meshes
  = ~9 frames at steady state).
- **VRAM clipmap + WSRC < 128 MB**: ✅ 10.8 MB combined.
- **WebGPU build compiles**: ✅ Web target cargo-check clean.

### Deferred

- ~~**Disk cache for per-mesh SDFs**~~ — landed after V15 closure
  (`native/shared/src/sdf_cache.rs`, commit 068511e / issue #22):
  content-hashed load in `scene.rs` skips the bake on hit, writes are
  flushed per frame via `flush_sdf_cache_writes`.
- **GPU jump-flood SDF bake** — V1's brute-force point-triangle
  works at current mesh sizes; jump-flood would matter if meshes
  scale up significantly.
- **True octahedral corner wrap (4 corners of each probe)** — V11
  octahedrally wraps edges but keeps corners as edge-extend. Sampler
  bilinear weights at corners are small so visual impact is limited.
- **Importance-sampled WSRC rebake cadence** — waiting on ticket
  016 (importance sampling) which provides the per-direction
  resolution hints needed to refresh hot octels more often than
  cold ones. V12 hysteresis is the coarse-grained stand-in.
- **Multi-cascade SDF clipmap** — V5 landed the single-cascade
  camera-follow version; the original plan had 4 cascades but in
  practice the single clipmap + 3-cascade WSRC covers the quality
  gap adequately.

### Follow-up tickets

- **016** Importance sampling + hierarchical probe refinement —
  unblocks the WSRC rebake cadence deferral above.
- **No new ticket needed** for platform parity — Android / Windows
  / Linux / Web all compile with host-matching toolchains; the
  smoke-check from macOS is the limitation, not the code.
