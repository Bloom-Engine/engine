# 007a — Lumen screen probes (SW trace, Hi-Z)

**Effort:** 2-3 days · **Expected gain:** SSGI 2×+ faster · **Status:** open

Supersedes the old ticket 007. Phase 1a of the [Lumen roadmap](lumen-roadmap.md).
Runs in parallel with 007b; both depend on 007-prep.

## Problem

Current SSGI is a per-pixel march: 8 cosine-weighted samples × 14 logarithmic
march steps = **112 iterations per half-res pixel**, ~24 ms/frame on Sponza at
800×450 half-res (measured: 23.6 fps with SSGI on, 55.1 fps off → 42.36 ms vs
18.15 ms).

UE5 Lumen replaces this with a Screen Space Radiance Cache (SSRC): one probe
per 16×16 pixel block, 32 rays/probe/frame into an 8×8 octahedral atlas,
spatial + temporal filtering, bilateral per-pixel reconstruction. At 800×450
half-res that's 50×29 = 1 450 probes × 32 rays = 46 400 rays/frame vs the
current ~2.9 M — ~60× fewer rays.

## Approach — file-by-file

### Shaders — `native/shared/src/renderer/shaders.rs`

Delete `SSGI_SHADER_WGSL` (lines 2219-2405) and `SSGI_TEMPORAL_SHADER_WGSL`
(lines 2412-2496). Add:

- `PROBE_HELPERS_WGSL` — `oct_encode(dir) -> vec2<f32>`, `oct_decode(uv) -> vec3<f32>`,
  `view_pos_from_depth(uv, depth, inv_proj)`, `reconstruct_view_normal(view_pos)`
  via `dpdx`/`dpdy`, shared `hiz_march(...)` helper.
- `SSGI_PROBE_PLACE_WGSL` — compute, 8×8 workgroup, one invocation per probe.
  Samples depth at a Halton-jittered UV inside the tile, skips sky, writes
  `ProbeHeader` to storage buffer.
- `SSGI_PROBE_TRACE_SW_WGSL` — compute, 8×8 workgroup = 64 lanes per probe.
  Each lane produces one octahedral-texel radiance value via Hi-Z march into
  the HDR buffer. Inner loop mirrors the existing SSGI march (fine start,
  geometric growth, off-screen termination, thickness check, firefly clamp
  at luma=10), but steps up the Hi-Z pyramid when step size exceeds the
  mip's texel footprint so late steps cost a single tap.
- `SSGI_PROBE_FILTER_WGSL` — compute. 3×3 bilateral blur in probe-XY per
  octahedral slice, weighted by probe depth + normal.
- `SSGI_PROBE_TEMPORAL_WGSL` — compute. Per-probe reprojection through
  `velocity_rt`; neighborhood min/max clamp; `α = 0.25` (4-frame window);
  disocclusion-forced `α = 1.0` on depth mismatch > 10%.
- `SSGI_PROBE_RESOLVE_WGSL` — fragment, writes half-res `ssgi_rt`. Per
  pixel: sample 2×2 enclosing probes, bilateral-weight by depth + normal
  match, evaluate each probe's octahedral atlas along pixel normal,
  weighted sum.

### Renderer — `native/shared/src/renderer/mod.rs`

New members on `Renderer`:

- `probe_header_buffer: wgpu::Buffer` — `ProbeHeader { world_pos: vec4, normal_oct: u32, depth: f32, valid: u32, _pad: u32 }`, 32 B × 1 450 probes ≈ 46 KB, `STORAGE | COPY_DST`.
- `probe_radiance_textures: [wgpu::Texture; 2]` + views — 3D, Rgba16Float,
  `probe_grid_w × probe_grid_h × 64` (z is octahedral index).
- `probe_radiance_filtered: wgpu::Texture` + view — same shape.
- Five pipelines + layouts + `_bg_cache: Option<_>` entries mirroring the
  existing SSAO/SSGI pattern.
- `SsgiProbeParams { inv_proj, proj, view, inv_view, grid: vec4<u32>, params: vec4<f32> }`.

Remove: `ssgi_pipeline`, `ssgi_layout`, `ssgi_bg_cache`, `ssgi_temporal_pipeline`,
`ssgi_temporal_layout`, `ssgi_temporal_uniform_buffer`, and the two render-pass
blocks at mod.rs:5618-5727.

Add the five new compute/fragment passes in their place, in order:
placement → trace → filter → temporal → resolve. The resolve pass writes
`ssgi_rt_view` which TAA already reads — the downstream composite is
untouched (`ssgi_composite_view` selector at mod.rs:5732 stays).

Resize path (mod.rs:3870 neighborhood): rebuild probe textures when
half-res size changes; invalidate bind-group caches.

### Example — `examples/intel-sponza/main.ts` (optional)

Add `--probe-debug <mode>` flag → new FFI `bloom_ssgi_set_debug_view(u32)`
where 1=probe placement, 2=single-probe octahedral atlas, 3=raw radiance
pre-filter. Development aid only; land iff needed for visual review.

## Acceptance

- `./examples/intel-sponza/main --quality 3 --ssgi 1 --fps-only 300` ≥ **47.2 fps**
  (2× the 23.6 baseline).
- `./examples/intel-sponza/main --quality 0 --fps-only 60` hits 60 (regression guard).
- `--capture 30 /tmp/sponza_007a_after.png` vs baseline: indirect bounce on
  shaded column sides preserved, under-arch indirect preserved, no persistent
  fireflies after 4-frame convergence. SSIM ≥ 0.95 against baseline indirect
  contribution.
- Disocclusion test: rapid camera pan produces no probe ghosting lasting
  longer than ~4 frames.
- Compiles for all 7 platforms (CI matrix / `cargo check` per target).

## Notes

- Reuses the Hi-Z pyramid from ticket 002 (5 mips, R32Float, half-res). Do
  not build a second pyramid.
- No world-space normal G-buffer exists. Reconstruct normals from depth at
  probe placement only (once per probe), not per ray.
- Probe grid size is `ceil(half_w / 16) × ceil(half_h / 16)`. Recompute on
  resize.
- Probes that land on sky (depth ≥ 0.9999) flag `valid = 0` and all 64
  octahedral texels write zero; resolve skips invalid probes in its 2×2
  bilateral.
