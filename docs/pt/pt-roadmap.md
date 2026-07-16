# Path tracing — three tiers, one kernel

A real GPU path tracer for Bloom, built on the ray-query/TLAS infrastructure
that Lumen already maintains. Three tiers share one megakernel; they differ in
what a ray hit can know (Tier 2) and in how few samples we can get away with
per frame (Tier 3).

This is **in addition to** Lumen, never instead of it. Lumen remains the
default gameplay GI on every platform; path tracing is a render *mode* that
engages where the hardware makes it honest, and falls back cleanly where it
doesn't.

## Why this shape

wgpu 29 exposes **inline ray query only** — no ray-tracing pipelines, no
shader binding tables, no `traceRay`, no hit shaders. So the tracer is a
compute megakernel that owns its own bounce loop and material dispatch. That
is a constraint, but it is also the portable shape: the same WGSL runs on
DXR 1.1, Vulkan ray query, and Metal.

The CPU reference tracer (`tools/bloom-reference`) is the acceptance oracle
for all of this. Two independent implementations agreeing is the strongest
correctness signal available to us — and the GPU tracer inherits the
reference's conventions (GGX + Burley, NEE with MIS balance heuristic, ACES +
fixed exposure) so images are comparable number-to-number.

## Modes

| Mode | Who | What |
|---|---|---|
| `0` off | default | Lumen GI as today |
| `1` progressive | editor "final quality" view, stills, ground truth in-engine | 1 spp/frame accumulated indefinitely while the camera is still; reset on move/scene change. No denoiser — convergence is the denoiser. |
| `2` realtime | gameplay on RT-capable GPUs | 1 spp/frame, temporal reprojection accumulation + à-trous spatial filter; optional half-res. Honest label: denoised 1-spp path tracing, the same family as Quake II RTX. |

Mode is a runtime setting (`bloom_set_path_tracing`), toggleable per frame.
When PT is active, SSGI/SSR/GTAO are skipped (their output would be
overwritten; skipping banks their cost). Shadow cascades still render — mode 2
reuses them nowhere, but the card-light pass (Tier 1 hit shading) does.

## Tiers

### Tier 1 — PT-1: progressive megakernel (correct diffuse transport)

Primary hit reconstructed from the **G-buffer** (free, and sharper than traced
primaries). Then per pixel per frame:

- NEE to the **sun**: sample the solar disc (~0.53°), one `rayQuery` shadow
  ray. Sky misses exclude the disc so BSDF samples cannot double-count it.
- NEE to **point lights**: sample one light per bounce, inverse-square +
  range falloff matched to the raster path.
- **Diffuse bounce**: cosine-weighted hemisphere; hit shading from the Mesh
  Card albedo atlas (textured) or flat instance albedo — the same
  `InstanceGiData` the Lumen HW trace reads. Emissive picked up at hits.
- Russian roulette from bounce 3; max 8 bounces (progressive) / 2 (realtime).
- Accumulate into rgba32float; reset when the view matrix moves beyond
  epsilon or `tlas_version` bumps.

Known honest limits at this tier: flat per-mesh hit normals and
card-resolution albedo → no sharp specular at secondary hits, no glass.
Correct, converged **diffuse** transport with real shadows and real
multi-bounce colour bleed.

### Tier 2 — PT-2: real materials at the hit

- **Geometry megabuffer**: at BLAS build, mesh positions/normals/UVs are also
  appended to one shared storage buffer; instances carry
  `{vtx_base, idx_base, material_id}`. One buffer — deliberately NOT
  `binding_array<storage>` — so geometry fetch works wherever ray query does.
- Hit shading fetches the triangle, interpolates normal + UV.
- **Albedo textures** through `binding_array<texture_2d<f32>>`, gated on
  `TEXTURE_BINDING_ARRAY` (+ non-uniform indexing); falls back to card
  albedo where absent. DX12 with DXC and Vulkan both qualify.
- **GGX specular lobe** with lobe selection + MIS, mirroring
  `bloom-reference/src/tracer.rs` so the two tracers stay comparable.
- Emissive from material data (VFX/muzzle flashes become real light in PT).

### Tier 3 — PT-3/PT-4: gameplay

- **PT-3**: temporal reprojection accumulation using the TAA velocity buffer
  (disocclusion via depth/normal test, neighbourhood clamp, capped alpha),
  then 3 à-trous iterations guided by depth + normal, firefly clamp.
  Optional half-res trace + bilateral upsample (SSGI resolve pattern).
- **PT-4**: ReSTIR DI — per-pixel weighted reservoirs over light candidates,
  temporal reuse with M-cap, spatial reuse over 3–5 neighbours, one shadow
  ray for the winner. **Experimental flag until validated against
  bloom-reference.** With today's handful of analytic lights (the engine
  caps point lights at 16; arena_02 uses 5) plain NEE is nearly as good;
  the payoff arrives when emissive particles/muzzle flashes become
  lights (Tier 2). Do not oversell it before then.

An ML denoiser is explicitly out of scope: nothing in wgpu runs vendor
denoisers, and a hand-rolled network is not happening. À-trous + temporal is
the honest ceiling here — it will look like early-RTX-era PT, not Cyberpunk
overdrive. That is the "as far as reasonable" line.

## Platform / fallback matrix

| Situation | Behaviour |
|---|---|
| `ray_query=true` (DX12+DXC, Vulkan RQ, Metal) | all modes available |
| `ray_query=false` | `bloom_set_path_tracing` is a no-op → Lumen; boot line + `bloom_path_tracing_supported()` say why |
| `TEXTURE_BINDING_ARRAY` absent | Tier 2 textures fall back to card albedo; everything else unaffected |
| Web | never (no WebGPU RT); permanently Lumen SW |

On Windows, `ray_query=true` is no longer automatic on capable hardware:
since 66dad5b the `EXPERIMENTAL_RAY_QUERY` device feature is only requested
when the process is launched with `BLOOM_HW_GI=1`, `BLOOM_PT`, or `--pt`
(`native/windows/src/lib.rs`). Without one of those, the first row behaves
as `ray_query=false` even on an RT-capable GPU. macOS/Linux still request
ray query by default (`BLOOM_FORCE_SW_GI` disables it).

## Verification protocol (per the hard lessons)

- **Binary-colour probes first**: every new pass proves it executes by
  writing an unmistakable solid colour before any real math ships. No
  image-diff conclusions under auto-exposure — PT comparisons run with
  manual exposure, same as reference parity.
- **Convergence check**: progressive mode screenshot at ~2 s vs ~15 s; the
  later one must be visibly smoother, not merely different.
- **Ground truth**: match camera + fixed exposure, render the same scene in
  `bloom-reference`, compare with `bloom-diff` (RMSE/SSIM). Acceptance for
  Tier 1: "close, differences explainable by flat hit normals / card-res
  albedo." Tier 2 tightens that.
- **Perf on a fight, not the title screen** (perf round-3 lesson), and any
  perf win larger than the change justifies = deleted geometry — screenshot
  3×.

## Tickets

| Ticket | Scope |
|---|---|
| PT-1 | Progressive megakernel, accumulation, FFI mode plumbing, shooter toggle |
| PT-2 | Geometry megabuffer, interpolated hit attributes, texture binding array, GGX+MIS, emissive hits |
| PT-3 | Realtime mode: temporal reprojection + à-trous, half-res option, perf gate |
| PT-4 | ReSTIR DI (experimental flag) |
| PT-5 | Settings/editor/gameplay integration, fallback matrix, reference-diff CI hook |
