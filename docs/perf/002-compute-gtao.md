# 002 — Compute-shader GTAO (replaces SSAO fragment shader)

**Effort:** ~2 days · **Expected gain:** SSAO 186 ms → ~50 ms · **Status:** open

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
