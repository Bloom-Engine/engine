# 005 — Depth prepass for main HDR pass

**Effort:** ~1 day · **Expected gain:** main_hdr 17 ms → ~8 ms · **Status:** open

## Problem

The `main_hdr_pass` renders Sponza's geometry with full PBR shading into 4
MRTs. The scene has heavy overdraw — arches, columns, and back walls overlap,
so every pixel gets shaded 3-5 times on average. Each redundant fragment runs
normal mapping, IBL sampling, PCF shadow lookup, and the full lighting
calculation before being rejected by depth.

Modern GPUs can do "early-Z rejection" automatically when the shader has no
`discard`/`alpha-to-coverage`, but that only helps when geometry is submitted
*front-to-back*. Bloom draws in handle order — not sorted — so overdraw is
full cost.

## Proposed approach

Classic **Z-prepass**. Before `main_hdr_pass`:

1. **Add a depth-only pipeline** (`scene_depth_pipeline`):
   - Same pipeline layout as `scene_pipeline` (reuses its bind groups).
   - Same vertex shader `vs_main_scene` (for correct transform + jitter).
   - **No fragment shader** (wgpu supports depth-only pipelines).
   - No color targets.
   - Depth state: write enabled, `Less`.
2. **Add a depth-prepass render pass** before `bloom_hdr_pass`:
   - Loads/clears depth to 1.0.
   - Iterates `scene.nodes` the same way `scene.render()` does (in-view
     frustum only — ticket 005 of this list means all scene nodes already have
     the flag).
   - Draws each node with the depth-only pipeline.
3. **Change `bloom_hdr_pass`'s depth attachment** from `LoadOp::Clear(1.0)` to
   `LoadOp::Load`.
4. **Switch scene draws in `main_hdr` to use `DepthCompare::Equal`** and
   disable depth writes. Requires either a second scene_pipeline variant
   (`scene_pipeline_equal`) or a dynamic pipeline state (not supported in
   wgpu). Go with two pipeline objects.
5. **Sky pass is unchanged** — it already writes depth = 1.0 with LessEqual or
   similar. At pixels where the prepass wrote depth < 1.0 (geometry), sky
   fails the test and sky's fragment shader doesn't execute. Bonus win.
6. **Immediate-mode 3D pass** keeps its current depth state (write Less) —
   immediate-mode games don't benefit from prepass but don't regress either.

## Trade-offs

- Vertex work doubles for scene geometry (once in prepass, once in main). On
  Sponza vertex shader is cheap (the PBR cost is all in the fragment stage),
  so this is a clear net win.
- For games with cheap fragment shaders (lots of unlit 2D or simple 3D), the
  prepass is a net loss — make it toggleable via `setEarlyZEnabled(bool)` on
  the TS API.

## References

- "GPU Gems 3 — Chapter 19" for basic Z-prepass rationale.
- UE4's `DepthPrepass` in the renderer source.

## Acceptance

- `./main --fps-only 300` default case: ≥ 1.5× FPS improvement on Sponza (e.g.
  2.8 → 4.2 fps at full quality; combined with other tickets, contributes to
  the 60 fps goal).
- Screenshot: identical to baseline — the PBR output must match pixel-for-pixel
  when depth test is Equal.
- `--quality 0 --fps-only 60` unchanged (prepass does nothing when there's no
  scene geometry).

## Notes for the implementer

- wgpu pipeline with no fragment state and no color targets is legal —
  `fragment: None`.
- Use the same vertex layout as `scene_pipeline` (`Vertex3D::desc()`).
- The prepass's view matrix *must* include any TAA jitter the main pass uses,
  or depth values won't match and the Equal test will reject everything.
- When frustum-culling is combined with prepass: use the same
  `in_view_frustum` flag for both.

## Files likely to change

- `native/shared/src/renderer.rs` — new `scene_depth_pipeline`, new
  `scene_pipeline_equal`, new prepass block in `end_frame_with_scene`.
- `native/shared/src/scene.rs` — add a `render_depth_only()` method mirroring
  `render()`.
