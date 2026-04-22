# 012 — Cache the 5 remaining per-frame `create_bind_group` calls

**Effort:** ~0.5 day · **Expected gain:** ~15-30 µs CPU · **Status:** deferred

## Problem

Commit `95da6af` cached 4 static bind groups (SSAO, SSAO-blur, SSR, SSGI).
The audit that session ran (`docs/perf/README.md` bisect) identified 14
`create_bind_group` calls per frame in `end_frame_with_scene`, of which 8
were cacheable and 6 were genuinely per-frame (due to rotating RT views).

Five of the eight cacheable ones weren't done:

| Bind group | Line | Complication |
|---|---|---|
| `motion_blur_bg` | ~9095 | Input view is conditional (taa / dof / hdr depending on which are enabled) |
| `sss_bg`         | ~9147 | Same conditional-input issue |
| `dof_bg`         | ~9046 | Same |
| `scene_compose_bg` | ~8983 | References `ssgi_composite_view` which alternates with `ssgi_history_idx` every frame |
| `composite_bg`   | ~9320 | References `exposure_views[dst_idx]` which ping-pongs |

## Proposed approach

Two patterns:

1. **Conditional-input bind groups** (motion_blur, sss, dof): key the cache
   on a small state bitmask reflecting which upstream passes are active.
   Usually 4 or 8 possible combinations; pre-build all on first use or build
   lazily and cache.

2. **Ping-pong bind groups** (scene_compose, composite): cache TWO bind
   groups, one for each ping-pong state. Index by `ssgi_history_idx % 2` /
   `exposure_dst_idx % 2`. Alternate reads without rebuilding.

All existing invalidation (resize, etc.) must clear these caches same as the
existing four.

## Acceptance

- Profiler shows `render_total` CPU drops by 15-30 µs on a default frame.
- No visual regression (run the standard screenshot diff).
- Resize still works correctly (caches invalidated).

## Notes for the implementer

- Pattern to follow: existing `ssao_bg_cache: Option<BindGroup>` pattern in
  `renderer.rs`, with `self.ssao_bg_cache = None` in `resize()`.
- Where inputs rotate (ping-pong), use `[Option<BindGroup>; 2]` and index by
  the relevant counter.
- The conditional-input caches may want `Option<(StateMask, BindGroup)>` so
  you can tell when the cached version is stale vs when it was never built.

## Files likely to change

- `native/shared/src/renderer.rs` — add 5 more cache fields, build/read
  them in `end_frame_with_scene`, invalidate in `resize()`.

## Deferred — prototype notes

A V1 prototype landed all five caches as `Option<(u32, wgpu::BindGroup)>`
fields, keyed on small state enums:

- `scene_compose_bg_cache`: `ssr_history_idx` when SSR is on, else 2 for the
  no-SSR path (three entries total).
- `dof_bg_cache`: `taa_dst_idx` when TAA on, else 2 (three entries).
- `motion_blur_bg_cache`: 0=DoF, 1/2=TAA (by ping-pong), 3=hdr (four entries).
- `sss_bg_cache`: 0=MB, 1=DoF, 2/3=TAA, 4=hdr (five entries).
- `composite_bg_cache`: `composite_src_variant | (exposure_dst_idx << 4)`
  covering the same composite_src branch chain (six variants) times the
  exposure ping-pong (12 entries).

All five invalidated alongside the existing ssao / ssr caches in
`resize()`. The pattern built and passed `--quality 3 --ssgi 1 --fps-only
300` at 60 fps vsync, but on interactive launch the window rendered
entirely black. Most likely cause: one of the cached bind groups bound
a view (probably in the composite chain) whose contents had not yet
been written by the current frame's pass on the first frame the key
appeared — i.e. the key covered the *input selection* but missed an
implicit frame-order dependency. Re-opening would need either:

1. Per-pass screenshot diff after each cache is wired, so a regression is
   caught at the first broken cache rather than the composite chain.
2. A more aggressive key that includes the "was this view written this
   frame?" bit, or just the pure `[Option<BindGroup>; N]` form so the
   prior frame's bg stays resident rather than getting evicted when the
   state flips and back.

Given the ticket's own best-case 15-30 µs CPU gain and the perf-README
rule that "Sponza is GPU-bound, not CPU-bound. Don't chase CPU micro-
optimizations expecting FPS improvement," this is parked until a scene
shows up that's actually CPU-bound on bind-group rebuilds. At that point
the prototype should be the starting point, with the screenshot-diff
protocol baked into the per-pass rollout from the start.
