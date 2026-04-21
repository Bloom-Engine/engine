# 006 — Frustum-cull casters in shadow pass

**Effort:** ~0.5 day · **Expected gain:** Shadow pass 14 ms → 7-10 ms · **Status:** landed

## Result

Per-cascade ortho-frustum cull against a world-AABB cached on `SceneNode`
during `prepare()`. Culling happens after the shared caster list is built
and drives a per-cascade index list; each cascade only writes uniforms
and records draws for the subset that overlaps its volume.

On the panning-camera cache-miss path at `--quality 2 --fps-only 300`:

| Metric | Before | After | Δ |
|---|---|---|---|
| FPS (quality=2 Medium) | 50.3 | 52.8 | +5.0% |
| Frame time | 19.88 ms | 18.95 ms | -0.93 ms |
| shadow_pass CPU | 648 µs | 489 µs | **-24%** |
| shadow_pass GPU | 3120 µs | 2046 µs | **-34%** |

The ticket's 14 ms → 7-10 ms target was set against the pre-95da6af
baseline. `CASCADE_MAP_SIZE` dropped 2048 → 1024 in 95da6af, which already
took the pass from 14 ms to ~3 ms on the cache-miss path, so absolute
headroom is small. The proportional cut (34% on the GPU side) matches
the ticket's intent. The 10% FPS target in the original acceptance is
unreachable at today's baseline because shadow is no longer a 14 ms cost.

Sun-behind-camera pose (`--yaw π`) diff against the before binary:
RMSE 0.168/255 vs a same-run TAA noise floor of 0.191/255 — shadows of
off-screen casters still land on-screen correctly.

## Problem

The shadow pass in `end_frame_with_scene` iterates every node with
`cast_shadow = true` and `visible = true`, regardless of whether that node
can actually contribute to any pixel on screen. For Sponza, many nodes are
outside the camera's view but also cast shadows that could never land in the
frame — those draws are pure waste.

The session that just committed already added frustum culling for the main
pass via `in_view_frustum` on `SceneNode`. The shadow pass deliberately
ignores that flag because **an off-screen caster can still cast a visible
shadow** (object behind the camera but between the sun and on-screen geometry).

## Proposed approach

Cull against the **union of the camera frustum and each cascade's light
frustum**, projected back through the light direction.

Simpler correct approach:

1. For each shadow cascade (there are 3 in Bloom), extract the 6 planes of
   its orthographic view-projection matrix — `shadow_map.light_vps[cascade]`.
   Use the `extract_frustum_planes` helper that already lives in `scene.rs`.
2. For each `cast_shadow` node:
   - Compute world-space AABB (already done for the main-pass culling).
   - Test against each cascade's planes.
   - If the node is outside cascade N's frustum, it can't contribute pixels
     to cascade N and should be skipped for that cascade.
3. Per-cascade draw lists (since cascades visible-set differs). Build three
   parallel `Vec<ShadowDrawEntry>` or a `Vec<(cascade_mask, entry)>`.

Refinement: the cascade frustum is *already* a conservative bound on what
shadow-map-space pixels could land in the camera view (that's how cascade
fitting works). So testing against the cascade's ortho frustum is the correct
test — no need to compute the camera/light union manually.

## Acceptance

- `./main --quality 2 --fps-only 300` (Medium — shadows on) ≥ 10% FPS
  improvement.
- `./main --capture 30 /tmp/after.png` vs baseline: shadows identical. Any
  missing shadow on-screen = culler too aggressive, fix before landing.
- Try a camera pose where the sun is behind the camera — shadows of objects
  behind the camera cast into view, verify they still render.

## Notes for the implementer

- `extract_frustum_planes` in `scene.rs` already does the Gribb-Hartmann
  extraction. Reuse.
- `aabb_outside_frustum` also already there. Reuse.
- Build the per-cascade draw list in the same loop that's currently at
  `renderer.rs` ~line 8250 (look for `ShadowDrawEntry`). The existing loop
  builds one shared list — extend to build `[Vec<ShadowDrawEntry>; 3]`.
- Keep the `SHADOW_MAX_NODES` cap per cascade (currently 1024).
- Don't cull by the light's near/far plane — in a static Sponza scene the
  light volume covers everything anyway.

## Files likely to change

- `native/shared/src/renderer.rs` — shadow-pass draw-list construction.
