# PT-5 — gameplay integration, fight-scene perf, specular NEE (+ PT-4 ReSTIR)

Status: **landed** (2026-07-14, 760M / DX12+DXC, 4K @ renderScale 0.75).

## Specular NEE (closing the PT-2 gap)

`direct_light` now evaluates a GGX highlight (`nee_spec`: same D/F/V
terms as `sample_brdf`) for BOTH the sun and the sampled point light,
at EVERY path vertex. The shadow ray was already paid for by the
diffuse term, so this is pure math cost. No double counting: analytic
lights cannot be hit by BSDF rays, and sky misses exclude the sun disc.
The old primary-only sun specular block is deleted — one code path.

## Progressive + combat: skip, don't burn

`tlas_version` bumps every frame during combat (enemy transforms).
Progressive resets on it (correct — its accumulation assumes a static
scene) but used to still DISPATCH: full-res trace cost, raster on
screen. The reset now also skips the dispatch, same as the camera-moved
case. Measured: progressive during a fight now costs raster + ~0
(15.1 ms vs 14.9 ms GPU).

## Fight-scene perf protocol (perf-round-3 rule: measure a FIGHT)

Shooter PERFTEST mode-1 harness (auto-start, immortal player, scripted
kills marching the waves), 60-frame windows, profiler-free windows
reported. 760M, 4K output, renderScale 0.75, wave 0-1 combat:

| Mode | Light combat (≤3 alive) | Heavy brawl (4-6 alive) | GPU avg |
|---|---|---|---|
| Raster (PT off) | ~45 fps | 32-36 fps | 14.9 ms |
| RT (realtime PT) | ~27.5 fps | 20-23 fps | 33.3 ms |
| PROG (progressive) | ~45 fps (raster shown) | 32-36 fps | 15.1 ms |

RT's fight cost is ~18 ms GPU on this iGPU — a deliberate quality
trade the settings row lets the player make. PROG is free during
combat by construction (see above) and converges when things go quiet.

## Settings / fallback matrix (verified on hardware)

- Shooter video menu row **PATH TRACING: OFF / QUALITY (STILLS) /
  REALTIME** — persisted in settings.json (`video.pathTracing`),
  applied at boot, live-applied on change, cycled by F9 and the `--pt`
  CLI flag; all four surfaces share one settings value.
- `ray_query=false` → the row renders `N/A (NO RAY TRACING)` and the
  value is inert (`isPathTracingSupported` gate); boot line says why.
- Mobile profile never applies PT (no headroom, no settings screen).
- `TEXTURE_BINDING_ARRAY` absent → card-albedo hit shading fallback
  (unchanged since PT-2).
- Reference-diff CI hook: **deferred, documented**. The game scene
  cannot reach number-parity with `bloom-reference` while skinned
  enemies are absent from the TLAS (accepted PT-2 gap); the reference
  oracle remains the tool for engine test scenes. Revisit when a CI
  runner exists for GPU work.

## PT-4 — ReSTIR DI (experimental, BLOOM_PT_RESTIR=1)

Landed per the roadmap's own framing: the architecture for the day
emissive particles/muzzle flashes become lights — NOT a win today
(this game has ≤16 analytic lights; arena_02 has 5).

- RIS over 8 uniform candidates, target = luminance of the unshadowed
  diffuse+specular contribution at the shading point.
- Temporal reservoir reuse through the SVGF reprojection (shared
  rp_* basis), M-capped at 20×, target re-evaluated at the CURRENT
  shading point — no geometric bias; visibility is never folded into
  the reservoir, so no visibility-reuse bias either. Spatial reuse
  deferred with the many-light content that would justify its bias
  handling.
- One shadow ray for the winner; bounce vertices keep plain NEE
  (reservoirs are per-primary-texel).
- Reservoir ping-pong buffers (bindings 20/21) raised the kernel to 9
  storage buffers — `max_storage_buffers_per_shader_stage` (default 8)
  is now requested from the adapter at device creation on both
  platforms.
- Validated at parity against plain NEE: A/B stills differ by 40/765
  mean — inside this live scene's ~38-50 run-to-run noise floor — with
  no energy shift and no cost delta (26 vs 27 fps). Realtime mode
  only; progressive stays pure NEE (it is the in-engine oracle).

## Tickets closed

PT-4 (experimental flag, as scoped), PT-5. The three-tier roadmap
(docs/pt/pt-roadmap.md) is complete.
