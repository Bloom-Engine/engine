# 005 — Depth prepass for main HDR pass

**Effort:** ~1 day · **Expected gain:** main_hdr 17 ms → ~8 ms · **Status:** deprioritized (see Findings, 2026-04-21)

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

## Findings (2026-04-21, headless Sponza baseline)

Post-mortem after a prototype attempt: the ticket as originally scoped does
not pay off on the current pipeline, and the approach has correctness gaps
that need a larger design change to fix. Summary:

**Headroom collapsed.** The ticket's 17 ms → ~8 ms `main_hdr_pass` budget
predates ticket 001. After 001 (TSR half-res) the pass is already ~2 ms
GPU on the benchmark scene (300-frame default `--fps-only`, Intel Sponza).
Best-case rejection saves ~1 ms, which is ~10 % of a 9.4 ms frame, not 1.5×.
Prototype measured **+3–5 % FPS** at full quality (104.7 → 108.2 fps across
3 runs), inside the run-to-run noise band.

**Correctness failed on Sponza.** Driving scene draws with
`DepthCompare::Equal` (or `LessEqual`, no depth write) after a `Less`-write
prepass produced visibly different output — ~11.9 % of pixels differed from
baseline by ≥8/255, concentrated on the central column and back-wall meshes
(see investigation screenshots in the session log). The same run-to-run
variance between two identical prepass runs was ~0.07 %, so this is a real
rendering bug, not temporal-stochastic noise.

Two root causes, both structural:

1. **Sky pass orchestration.** The baseline sky pipeline is `Always + write`
   and runs first inside `main_hdr_pass`, laying down background color +
   depth=1.0 at every pixel. With a prepass wanting Load-depth afterwards,
   the sky clobbers the prepass depth. Switching sky to `LessEqual +
   no-write` is correct for depth but means sky color only lands at
   background pixels — any scene fragment that then fails the main pass's
   `Equal/LessEqual` (even for benign reasons) blends against the black HDR
   clear instead of the sky, producing dark artefacts. The three-pass
   workaround (sky → prepass → main-with-load) fixes this but splits a
   single render pass into three.
2. **Coplanar / layered scene geometry.** Sponza uses multiple overlapping
   meshes per surface (marble base + decal overlays). Under `Less + write`
   the first-drawn wins and subsequent coplanar surfaces fail Less strictly.
   Under `Equal` / `LessEqual` they all pass, so alpha blending composites
   previously-rejected decals into the final image — visually identical
   pixels get different colors than the baseline. This cannot be fixed with
   a depth bias or `@invariant` (both were tried).

There are also secondary issues a follow-up would need to address:

- The scene graph's `PbrMaterial` carries no `alpha_cutoff` or blend-mode
  flag. A correct prepass would need per-node opacity classification to
  skip alpha-mask geometry (whose discarded fragments must not write
  depth).
- `bloom_scene_attach_model` (macOS FFI) doesn't propagate the glTF
  material's alpha mode onto the scene node, so the runtime can't even
  detect it.

## Recommendation

Keep the ticket open conceptually but reorder: the structural wins now
come from tickets 006 (shadow-pass culling) and 007 (Lumen-style SSGI).
Revisit 005 only if a future profile shows `main_hdr_pass` regressing back
above ~5 ms — at which point the correct design is Unreal's pattern:

1. Per-material opacity flag on scene nodes, propagated from glTF.
2. Prepass with a *lightweight* alpha-test fragment shader that samples
   only base-color alpha and discards below cutoff.
3. Decal / layered-overlay meshes explicitly skip the prepass and render
   last in a dedicated depth-less pass.

That's a multi-day refactor, not the ~1 day this ticket originally
estimated. Not worth it while the frame budget is dominated by bloom /
TAA / SSGI / final-composite passes (the real targets of tickets 007 and
010).
