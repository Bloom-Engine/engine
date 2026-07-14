# PT-6/7/8 — skinned TLAS, real motion vectors, correctness oracle

Status: **landed** (2026-07-14, 760M / DX12+DXC).

## PT-6 — skinned meshes enter the TLAS

Skinned characters were ghosts to the tracer: skinning lived only in
the vertex shader, so no posed geometry existed for a BLAS, and the
TLAS was scene-nodes-only. Now:

- `cache_model_if_static` retains bind-pose CPU geometry (+ STORAGE on
  the VB) for skinned meshes; `draw_model_cached_skinned` registers
  each mesh as a `PtDynamicDraw` per frame.
- `rebuild_instance_data` appends a megabuffer window + instance entry
  per dynamic draw; a compute pre-skin pass (`PT_SKIN_WGSL`, the same
  palette blend as the raster VS) overwrites the window's
  position/normal with posed WORLD-SPACE data (the palette bakes
  placement → identity TLAS transform).
- Per-slot BLASes rebuild every frame FROM the megabuffer windows
  (`first_vertex`/`first_index`) — intersection and hit shading read
  the same bytes. The megabuffers gained `BLAS_INPUT` usage.
- The TLAS recreates when the total instance count outgrows the
  capacity it was built with (`tlas_created_cap`) — wave spawns grow
  the count mid-run, which the old check missed.

Lumen's HW probe trace consumes the same instances, so skinned
characters now contribute to GI too. Verified via the debug-6 traced
view: the skinned player renders with interpolated normals — the first
time any skinned mesh has appeared in a traced image.

## PT-7 — real skinned motion vectors + velocity-driven PT reprojection

Skinned draws wrote EXACTLY zero velocity (no previous-frame joint
palette existed anywhere; the cached path stamped `prev_mvp` with the
current VP). Enemies ghosted under TAA/TSR and invalidated their own
PT history every frame.

- The joint palette is double-buffered. `set_joint_matrices_scaled`
  takes a pairing key (the FFI anim handle); the previous palette for
  the same key stages in lockstep so both arenas share offsets. First
  sighting (spawn) pairs with itself = zero velocity, correct.
- The scene VS reconstructs last frame's world position from
  `joints_prev` (group 3 binding 1) and projects it through the EN-022
  velocity-reference VP — skeletal AND locomotion motion land in the
  velocity MRT.
- The PT kernel binds the velocity MRT (binding 22). `compute_reproj`
  follows per-pixel motion when non-zero (TAA's convention:
  `prev_uv = (uv.x − vel.x, uv.y + vel.y)`) and falls back to the
  camera `prev_vp` math otherwise. Moving skinned characters keep
  their SVGF history instead of resetting to 1 spp (debug-20 history
  heat: no rejection hole on the animating player).

## PT-8 — the correctness oracle

Two golden-image tests in `native/shared/tests/golden_render.rs`,
running the REAL engine headless on a ray-query device (skip cleanly
without one; on Windows `dxcompiler.dll`/`dxil.dll` must be loadable —
untracked local copies next to the crate, see .gitignore):

- **`pt_progressive`** — converged progressive mode (300 static
  frames) on a node scene. Catches transport regressions (BRDF energy,
  NEE, sky handling, accumulation math) as an image diff.
- **`pt_realtime_motion`** — realtime mode while the camera orbits.
  Catches reprojection/temporal regressions: a broken history (the
  prev_vp-transpose class that survived three human review rounds)
  floods the image with unconverged speckle, far past tolerance.

`BLOOM_UPDATE_GOLDEN=1 cargo test golden` regenerates; the strict
outlier gate stays global so looser mean tolerances cannot hide a
broken region.

**The oracle caught a real bug before it was even committed**: the
kernel seeded its RNG from `taa_frame_index`, which freezes when TAA
is off — the sample sequence froze and progressive accumulation
silently never converged (300 frames = the same image as 1). A player
disabling TAA in settings would have hit exactly this in-game. PT now
keeps its own rolling `pt_frame_index`.

The bloom-reference RMSE parity run remains a manual protocol (see
PT-5 doc): number-parity on the game scene is now *closer* (skinned
meshes are in the TLAS) but cutout foliage and card-resolution albedo
still make it a human-judged comparison, not a gate.
