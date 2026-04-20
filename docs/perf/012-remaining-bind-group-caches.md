# 012 — Cache the 5 remaining per-frame `create_bind_group` calls

**Effort:** ~0.5 day · **Expected gain:** ~15-30 µs CPU · **Status:** open

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
