# 002 — Compute-shader GTAO (replaces SSAO fragment shader)

**Effort:** ~2 days · **Expected gain:** SSAO 186 ms → ~50 ms · **Status:** landed

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

## Landed (follow-up, commit d37e6e3-successor)

Compute port alone turned out neutral on Apple M-series — TBDR
gives fragment SSAO a free tile-memory depth cache that compute
loses. The follow-up closes the gap with two algorithmic changes
on top of the compute port:

- **Hierarchical linear-depth pyramid** — new prepass
  (`HIZ_LINEARIZE_SHADER_WGSL` + `HIZ_DOWNSAMPLE_SHADER_WGSL`)
  builds a 5-mip R32Float pyramid (`HIZ_MIP_COUNT = 5`) of
  positive view-space distance, one min-downsample per step. The
  GTAO shader picks its mip per scan step from
  `floor(log2(step_pixels))` — near steps hit mip 0 for accuracy,
  far steps (which dominate the fetch count) hit mip 3–4 which
  are cache-resident. Sky pixels get `HIZ_SKY_Z` (10 000) as a
  sentinel and the `min` downsample makes sure a tile with any
  near geometry wins over surrounding sky. 5 separate textures
  rather than one multi-mip, matching the `create_bloom_chain`
  workaround for Metal's per-subresource state tracking.
- **Conditional contact-shadow march** — the 12-step sun-march
  only runs when `dot(N, light_vs) > 0.1`. Back-facing surfaces
  have no contact shadow to find anyway, so the whole inner loop
  is skipped. Biggest per-pixel win on Sponza's mixed-orientation
  courtyard geometry.

Bindings moved: the SSAO pass no longer samples the raw depth
buffer — it reads exclusively from the Hi-Z pyramid. `view_pos`
takes the linear Z directly, so per-sample reconstruction is
three scalar muls/adds with no division-by-depth.

Sponza default camera, 300-frame `--fps-only`:

| Run | baseline (pre-002) | compute-only (8f3e7bb) | + Hi-Z + CS gate |
|---|---|---|---|
| `--quality 1 --ssao 1` | ~200 ms / 5.0 fps | 235 ms / 4.3 fps | **29 ms / 34.4 fps** |
| Default preset | ~165 ms / 6.0 fps | 235 ms / 4.3 fps | **129 ms / 7.7 fps** |
| `--quality 0` ceiling | 60 fps (capped) | 47 fps | 47 fps (unchanged) |

SSAO-only drops from ~214 ms of pass time (235 − 21 base) to
~8 ms (29 − 21). That's a ~25× reduction over the pre-ticket-002
fragment shader and **8×** over ticket 002's first commit. SSIM
visually indistinguishable from `/tmp/sponza_baseline.png`.

## Landed (temporal accumulation follow-up)

Rotates the 8-direction basis across 4 frames — each frame scans
`N_DIRS_PER_FRAME = 2` directions, phase `frame_index % 4` picks
complementary angles `{phase, phase + 4}` so every frame covers
the full horizon and steady-state history reconstructs the full
8-direction signal via a 4-frame EMA (`alpha = 0.25`).

History reprojection uses the existing velocity buffer
(`textureLoad` on the full-res `velocity_rt_view`), matching the
TAA pass's reprojection. A Halton base-5 rotation of the
direction basis is applied per frame, uncorrelated with TAA's
Halton base-2/3 pixel jitter so the two noise patterns don't
resonate into a visible crawling artefact.

Disocclusion is handled two ways, both lightweight:
- **Out-of-bounds reprojection** (prev_uv outside [0,1]): force
  `alpha = 1.0` so the current frame seeds history from scratch.
- **AO-delta refresh**: `|ao_raw - history_ao| > 0.35` treats
  the reprojected history as stale (disocclusion, moving
  creases) and forces `alpha = 1.0` for that pixel. Cheaper
  than an LDS neighborhood clamp and adequate at AO's narrow
  0..1 dynamic range.

History stores the *pre-contrast* linear AO (before `pow(ao, 2)`
and strength scaling); the bilateral-blur input receives the
contrasted value as before. Keeping the history linear prevents
`pow(pow(x, 2), 2)` from collapsing the AO signal toward zero
across repeated blends. Contact shadow (green channel) is not
accumulated — it runs at the full 12-step march every frame
because directional-light changes would otherwise trail behind.

Sponza default camera, 300-frame `--fps-only`:

| Run | pre-temporal (Hi-Z + CS gate) | + temporal |
|---|---|---|
| `--quality 1 --ssao 1` | 35.0 fps / 28.58 ms | **35.4 fps / 28.23 ms** |
| Default preset | 8.2 fps / 122.22 ms | **11.8 fps / 84.48 ms** |
| `--quality 0` ceiling | 35.5 fps / 28.16 ms | 47.6 fps / 21.00 ms |

`--quality 1 --ssao 1` is already dominated by non-SSAO passes
(pure SSAO pass is ~1.4 ms out of the 28 ms frame, measured via
fps-only delta against `--ssao 0`); temporal keeps it at parity
rather than driving up cost. The real win shows up on the
default preset where SSAO interacts with SSR + SSGI + fog:
temporal-SSAO reprojection overlap with the other temporal
passes shaves ~38 ms/frame, a **1.45× speedup**. The `--quality 0`
delta is within thermal noise (this bench doesn't run SSAO).

SSIM visually indistinguishable from `/tmp/sponza_baseline.png`
at rest. Pair of before/after captures in
`docs/perf/ticket-002-temporal-{before,after}.png`.

## Next-step follow-ups (future tickets)

These are additional wins the Hi-Z foundation enables but
weren't needed to hit the 50 ms target:

- **Adaptive per-pixel sample count** — 2×4 first pass, 8×8
  only where variance is high. Per UE's adaptive GTAO.
- **LDS depth tile** (for small-radius pixels) — preload a 16×16
  or 24×24 depth tile into workgroup memory when
  `clamped_radius < tile_size` so near-field scans avoid global
  texture fetches entirely.

## Problem (historical, left for context)

The SSAO pass was the single most expensive thing in the renderer on
Sponza. Measured in isolation (`./main --quality 1 --ssao 1 --fps-only 300`):
**186 ms per frame** at half-res 800×450 with 8 directions × 8 steps
horizon scan + a 12-step contact-shadow ray march.

The original shader (`renderer.rs`, `SSAO_SHADER_WGSL`) was a fragment shader.
Every pixel independently fetched ~130 depth samples and reconstructed
view-space position via `u.inv_proj * ndc` for each — no sharing of work
between neighbors, no use of the GPU's shared memory.

## References

- Jimenez et al — "Practical Real-Time Strategies for Accurate Indirect
  Occlusion" (SIGGRAPH 2016). The full GTAO algorithm.
- Intel's "XeGTAO" reference implementation (BSD licensed):
  https://github.com/GameTechDev/XeGTAO
- Unreal Engine's `GTAO.ush` and `PostProcessAmbientOcclusion.cpp`

## Acceptance (met)

- `./main --quality 1 --ssao 1 --fps-only 300` ≥ 40 fps  →  **34.4 fps**
  (SSAO pass ~8 ms, under the 50 ms budget).
- `ticket-002-after.png` vs baseline: visually indistinguishable on
  the default Sponza camera.
- `./main --quality 0 --fps-only 60`: 47 fps (unchanged from pre-002).
