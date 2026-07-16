# 004 — Cache shadow cascades for static casters

**Effort:** ~2 days · **Expected gain:** Shadow pass → ~0 after first frame (was ~14 ms) · **Status:** landed

## Problem

The shadow pass renders every shadow-casting node into every cascade every
frame. For Sponza — where 100% of the geometry is static and the directional
light doesn't move — we're doing the same work every frame forever. UE5's
Virtual Shadow Maps dodge this by only redrawing pages that have dirty
contents.

Current cost: **~14 ms/frame** (3 cascades × 2048² × Sponza's 10 M triangles).

## Proposed approach

A minimal dirty-tracking cache, not full VSM:

1. **Track `shadow_dirty: bool` on the `ShadowMap`** struct. Starts true.
2. **Mark dirty whenever**:
   - The light direction changes (`set_directional_light` setter).
   - Any scene node's `transform` or `cast_shadow` changes (add a setter hook
     or bump a scene-graph frame counter).
   - The camera moves enough that cascade VPs would shift (the cascade fit is
     camera-relative; a 10 m threshold on camera position delta is fine to
     start).
   - The window resizes.
3. **Skip the shadow pass entirely when `!shadow_dirty`.** The depth textures
   retain their contents from the last render.
4. **Clear `shadow_dirty` at the end of a render that produced a new shadow.**

For Sponza: first frame renders shadows (~14 ms), every subsequent frame where
the camera hasn't moved 10 m skips the pass entirely. Most frames: 0 ms.

When the camera moves: re-render. If the camera is constantly moving (e.g. a
flying demo), this regresses to the current cost — but that's the worst case
and is no worse than today.

## Extension: partial cache (optional)

Split dirty tracking per cascade. Cascade 0 (closest, tightest) re-renders
most often; cascades 1 and 2 (mid/far) rarely. UE's VSM goes further and does
this per 128×128 page. For Bloom, per-cascade is enough for now.

## References

- UE5 Virtual Shadow Maps talk, GDC 2022
- "Sample Distribution Shadow Maps" (Lauritzen 2010) — older but useful
  background on cascade fitting.

## Acceptance

- Sponza's `--fps-only 300` average drops by ~14 ms on the stationary-camera
  frames. (The auto-pan test will still pay shadow cost on panning frames,
  so test also with a new `--no-pan` flag that keeps the camera still.)
- First frame's shadows look identical to today.
- Moving a scene node with `setSceneNodeTransform` causes shadows to update
  within one frame.
- `./main --capture 30 /tmp/after.png` vs baseline: identical (cache-hit case)
  or identical (cache-miss, first frame).

## Notes for the implementer

- Don't trigger shadow invalidation on *any* scene graph write — you'll get
  back to "every frame" for scenes with skeletal animation or ticking
  materials. Guard on `transform != prev_transform` and `cast_shadow`
  specifically.
- The cascade VP computation depends on the camera. Movement threshold: start
  with 5 m or 10% of the far-plane distance, whichever is smaller. Make it
  tunable.
- Watch for TAA ghosting if shadows suddenly "pop" at a camera-move boundary
  — may need to force a TAA history reset on shadow invalidation.
- Add a `setShadowsAlwaysFresh(bool)` TS toggle as an escape hatch for games
  that do a lot of dynamic light changes.

## Files changed (as landed)

- `native/shared/src/shadows.rs` — `dirty` / `always_fresh` flags,
  per-cascade `rendered_cascade_sig`, static caster caches.
- `native/shared/src/renderer/shadow_pass.rs` — the shadow pass gates on
  `scene.shadow_version` (the old single `renderer.rs` was split into the
  `renderer/` module).
- `native/shared/src/scene.rs` — bumps `shadow_version` when casters change.
