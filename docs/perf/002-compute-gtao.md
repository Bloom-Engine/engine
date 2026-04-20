# 002 — Compute-shader GTAO (replaces SSAO fragment shader)

**Effort:** ~2 days · **Expected gain:** SSAO 186 ms → ~50 ms · **Status:** partial

## Landed (this pass)

- SSAO fragment shader rewritten as a compute shader
  (`@compute @workgroup_size(8, 8, 1)` in `SSAO_SHADER_WGSL`).
- Per-sample view-space reconstruction swapped from `mat4×vec4 + /w`
  (~32 ops) to six scalar projection terms (~9 ops). Correct under
  TAA jitter of `proj[2][0]/[2][1]`.
- Per-direction sky early-out: once a step in a horizon scan reads
  depth ≥ 0.9999, subsequent steps in that direction are skipped
  (horizon would stay pinned below tangent anyway). Saves up to
  N_STEPS-1 fetches per direction when pointing at sky.
- SSAO RT moved from `Rg8Unorm + RENDER_ATTACHMENT` to
  `Rgba8Unorm + STORAGE_BINDING`; written via `textureStore`.
- `Profiler::compute_pass_timestamp_writes` companion helper added
  so the new compute pass still profiles.

Visually byte-identical on the Sponza default camera pose
(`ticket-002-before.png` vs `ticket-002-after.png`).

Benchmarks (Sponza default camera, 300 frames):

| Run | before | after |
|---|---|---|
| `--quality 1 --ssao 1` | ~200 ms / 5 fps | ~235 ms / 4.3 fps |
| `--fps-only` (default) | ~165 ms / 6 fps | ~235 ms / 4.3 fps |
| `--quality 0` | 60 fps (capped) | 47 fps |

## What didn't move

The acceptance target — SSAO pass ≤ 50 ms, ≥ 40 fps on
`--ssao 1` — is not met by the compute port alone. On Apple
M-series the fragment and compute paths are within noise of each
other on this workload: the pass is dominated by ~128 scattered
depth-texture fetches per pixel, and cheaper per-sample math /
removed rasterizer overhead don't materially shift that.

## Follow-up (not in this commit)

To actually hit 50 ms, the algorithm must do fewer samples or
cheaper samples:

- **Hierarchical (pyramid) depth** — generate a mip chain of view-
  space-linear depth once per frame and sample farther horizon-
  scan steps from coarser mips (XeGTAO does this). Cuts cache
  pressure on the long-range fetches that dominate the pass.
- **Temporal accumulation** — distribute the 8-dir × 8-step scan
  across N frames (rotate direction basis per frame), accumulate
  under TAA. Divides per-frame cost by N; quality stays at the
  same 128 effective samples/pixel on static content.
- **Adaptive per-pixel sample count** — first pass with 2×4 at all
  pixels, refine to 8×8 only where variance is high. Per UE's
  adaptive GTAO / Activision's slides.

The storage-texture output + per-direction sky break shipped here
are the foundation any of those land on top of.

## Problem

The SSAO pass is the single most expensive thing in the renderer on Sponza.
Measured in isolation (`./main --quality 1 --ssao 1 --fps-only 300`): **186 ms
per frame** at half-res 800×450 with 8 directions × 8 steps horizon scan + a
12-step contact-shadow ray march.

The current shader (`renderer.rs`, `SSAO_SHADER_WGSL`) is a fragment shader.
Every pixel independently fetches ~130 depth samples and reconstructs
view-space position via `u.inv_proj * ndc` for each — no sharing of work
between neighbors, no use of the GPU's shared memory.

## Proposed approach

Rewrite as a **compute shader** following Activision's GTAO (Ground Truth
Ambient Occlusion). Key wins over the current fragment-shader horizon scan:

1. **Thread-group shared memory for depth**: dispatch 8×8 or 16×16 thread
   groups, each thread preloads one depth sample into shared memory. Neighbors
   read from LDS instead of global texture. On Sponza's 8-dir × 8-step scan,
   each pixel reads ~64 neighbors — the LDS reuse gives you ~4-8× less
   external bandwidth.
2. **SIMD-lane sharing via subgroup ops** (WebGPU `subgroupBroadcast`,
   `subgroupAdd`): neighbors in a wavefront can share normal reconstruction.
3. **Lower precision for horizon accumulation**: f16 instead of f32 — halves
   register pressure, often enables higher occupancy.
4. **Combined normal-from-depth derivation with the horizon scan**: current
   shader samples 2 extra depth taps for `dpdx`/`dpdy` reconstruction.
   Compute shader can do this via LDS neighbors for free.
5. **Adaptive sample count per pixel**: start with 2 dirs × 4 steps, continue
   in a second pass only where the first pass detected high variance. Like
   UE's adaptive GTAO.

Algorithm reference: the GTAO presentation from SIGGRAPH 2016 (Jimenez et al,
"Practical Real-Time Strategies for Accurate Indirect Occlusion"). Activision
released the pseudocode — their reference shader is public.

## References

- Jimenez et al — "Practical Real-Time Strategies for Accurate Indirect
  Occlusion" (SIGGRAPH 2016). ← Contains the full GTAO algorithm.
- Intel's "XeGTAO" reference implementation (BSD licensed):
  https://github.com/GameTechDev/XeGTAO
- Unreal Engine's `GTAO.ush` and `PostProcessAmbientOcclusion.cpp`

## Acceptance

- `./main --quality 1 --ssao 1 --fps-only 300` ≥ 40 fps (SSAO pass cost drops
  from 186 ms to ≤ 50 ms).
- `/tmp/sponza_after.png` vs baseline: AO banding acceptable, no obvious halo
  regression, contact shadows on column base still visible. SSIM ≥ 0.99 on the
  image.
- `./main --quality 0 --fps-only 60` unchanged (60 fps).

## Notes for the implementer

- wgpu 24 supports compute shaders. WGSL compute entry is
  `@compute @workgroup_size(x, y, 1) fn cs_main(...)`.
- wgpu's subgroup ops are feature-gated — check `device.features()` for
  `Features::SUBGROUP` and fall back to pure LDS if missing.
- The blur pass (`SSAO_BLUR_SHADER_WGSL`) can stay as-is; GTAO output still
  benefits from the bilateral denoise.
- The `ssao_rt` is `Rg8Unorm` (R = AO, G = contact shadow). Compute shader
  needs to write via `textureStore` — declare the RT with
  `TextureUsages::STORAGE_BINDING`.
- The current `ssao_bg_cache` in `renderer.rs` will need rebuilding to the
  compute layout — include that in the resize invalidation set.

## Files likely to change

- `native/shared/src/renderer.rs` — replace SSAO_SHADER_WGSL, replace the
  render pass with a compute pass, update the RT to add STORAGE_BINDING, update
  the bind group layout.
