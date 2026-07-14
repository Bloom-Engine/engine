# PT-1 — Progressive path-trace megakernel (Tier 1)

Parent: [pt-roadmap.md](pt-roadmap.md). Requires: hardware ray query (merged
in #91), TLAS + `InstanceGiData` + Mesh Card atlases (Lumen tickets 007b/013).

## Scope

One compute megakernel (`renderer/shaders/pt.rs`, pass in
`renderer/pt_pass.rs`) that replaces the lit scene colour when
`pt_active() && pt_mode == 1`:

1. Primary hit from the G-buffer (depth → world pos, normal, albedo,
   rough/metal). Sky pixels (far depth) get sky radiance directly.
2. Bounce loop, max 8:
   - **NEE sun**: sample the solar disc (half-angle 0.265°), one shadow ray.
     Delta-light radiometry identical to the raster sun so brightness
     matches. Sky misses exclude the disc (no double count).
   - **NEE point lights**: pick one of the frame's lights uniformly,
     shadow ray, contribution / pdf. Same falloff as raster.
   - **Diffuse bounce**: cosine hemisphere. Hit shading = card-atlas albedo
     (textured) or flat `InstanceGiData` albedo + emissive — exactly what
     the Lumen HW probe trace reads at hits today.
   - Russian roulette from bounce 3 on throughput luminance.
3. Accumulate into an rgba32float buffer
   (`accum = (accum * n + sample) / (n + 1)`); write tonemappable HDR into
   scene colour. Reset `n` when the view matrix moves beyond epsilon or
   `SceneGraph::tlas_version` bumps.

While active: skip SSGI, SSR, GTAO dispatches (banked cost, and their
output would be overwritten). Shadow cascades keep rendering — the card
light pass needs them, and mode 0 must be switchable back per frame.

## Milestone gates (each = compile + screenshot)

- **M1**: kernel writes solid magenta gated on `pt_mode >= 1`. Proves
  pipeline, bind groups, TLAS binding, dispatch, scene-colour write, F9.
- **M2**: sky + sun NEE + 1 diffuse bounce, accumulation working — 2 s vs
  15 s screenshots must show convergence (less noise, not merely
  different).
- **M3**: full scope above. Point lights visible at night-ish exposure;
  colour bleed visible at the building base (the EN-023 acceptance
  criterion, finally measurable).

## Acceptance

- Convergence: variance visibly ↓ between 2 s and 15 s stills.
- Parity: same camera + fixed exposure in `bloom-reference`
  (`--spp 256 --bounces 8`) vs converged in-engine frame; `bloom-diff`
  RMSE reported in the PR. Differences must be explainable by flat hit
  normals / card-res albedo (Tier 2 closes those).
- Toggling F9 off restores the raster path with no residue.

## Status (2026-07-13, AMD 760M / DX12+DXC)

- **M1 passed**: `BLOOM_PT_DEBUG=5` magenta over the full frame, HUD +
  translucent water compositing on top, no validation errors.
- Depth convention confirmed standard-Z (`BLOOM_PT_DEBUG=1`: near dark,
  far bright) — `is_sky(depth >= 0.9999999)` is correct as written.
- **M2 passed**: full render at brightness parity with raster; static
  title camera at ~17 s vs ~50 s shows clearly reduced grass/canopy
  noise (convergence, not mere difference). Fresh-accumulation noise
  visible as expected right after a camera cut.
- Live mode switching verified: F9 → RT (EMA) mid-game, F9 → off
  restores the raster path bit-clean (SSGI/SSR resume, 13 → 46 fps).
- **M3 partially open**: point-light NEE + building-base colour bleed +
  `bloom-reference` RMSE not yet measured (needs a lit world + fixed
  exposure; folded into PT-2/PT-5 verification).

Implementation notes discovered on the way:

- `current_vp_matrix` carries the TAA Halton jitter (~1e-3 in the proj
  Z-coupling slots). The accumulation-reset check MUST compare an
  unjittered VP (`current_proj_matrix_unjittered × view`) or mode 1
  pins at 1 sample forever. The *jittered* `inv_vp` still feeds the
  kernel — primary rays must match the jittered G-buffer depth, and
  accumulating across jitters is free AA.
- TAA runs downstream of PT, so converged PT output gets a second
  temporal filter (soft distant foliage). Acceptable for Tier 1;
  PT-3 owns the proper answer (bypass/replace TAA under PT).
- Ray queries treat leaf-card cutouts as solid quads on bounce rays
  (slight over-darkening under canopies). Primary visibility comes
  from the G-buffer so silhouettes are unaffected. PT-2 material at
  hits can alpha-test via `RAY_FLAG_NO_OPAQUE` + candidate loop if it
  proves visible.
- Batch verification: `BLOOM_PT`/`BLOOM_PT_DEBUG` env seed the renderer
  directly. Careful with F12-capture scripts: the title screen treats
  the posted F12 as "press any key" and starts the game.

## Post-scriptum (2026-07-13, PT-2 bring-up): the transposed inv_vp

PT-2's numeric readback (BLOOM_PT_DEBUG=16/17) revealed that EVERY ray
this kernel generated since PT-1 was degenerate: `current_inv_vp_matrix`
is stored transposed relative to what WGSL's `M * v` computes, so the
unprojection collapsed all primary rays into one bundle. Light transport
(sun shadow rays, bounces) traced garbage while the image still looked
plausible — primaries come from the G-buffer and direct NdotL carried
the shading. Fixed by uploading the transpose (pt_pass.rs). Lessons,
paid for in full:

- The house rule exists for a reason: every other pass unprojects via
  linear-z + projection coefficients and never touches an inverted
  matrix. The PT kernel was the first `inv_vp` consumer, and the first
  casualty.
- "Looks plausible" is not a transport test. The M2 convergence gate
  passed on top of broken rays because accumulation mechanics are
  independent of ray correctness. The debug-13 t-vs-G-buffer sanity
  view (green/red/blue) is now the mandatory first check for any
  ray-generation change.
- HDR-scaled probe colours (×20-50 to defeat tonemapping) saturate
  fract-band signals into uniform blobs — three investigation probes
  lied because of this. When probes disagree with expectations twice,
  switch to numeric readback (debug 16/17) immediately.
- wgpu 29 DX12+DXC inline ray query is fully correct (validated with a
  standalone repro: fat vertex strides, same-encoder build+dispatch,
  multiple queries, engine-shaped bind groups — all fine). The naga
  helper-function codegen needs no patching.

## Known honest limits (by design, closed by PT-2)

Flat per-mesh hit normals; card-resolution albedo at hits; no specular
bounce; no glass; sky at miss is the analytic Lumen sky, not the LUT sky.
