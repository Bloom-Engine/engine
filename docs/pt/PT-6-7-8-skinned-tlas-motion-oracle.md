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

`BLOOM_UPDATE_GOLDEN=1 cargo test golden` regenerates. PT golden
updates are forbidden while a negative-control fault is active. The
comparison reports mean RGBA/RGB error, maximum channel error, SSIM,
and the fraction of *pixels* (not channels) whose RGB error exceeds
32/255. The 1% structural-outlier gate stays strict so a permissive
mean cannot hide a broken region.

**The oracle caught a real bug before it was even committed**: the
kernel seeded its RNG from `taa_frame_index`, which freezes when TAA
is off — the sample sequence froze and progressive accumulation
silently never converged (300 frames = the same image as 1). A player
disabling TAA in settings would have hit exactly this in-game. PT now
keeps its own rolling `pt_frame_index`.

### Deterministic hardware protocol

The PT tests make every stochastic and temporal input explicit:

- seed `0`, sample sequence start `0`, camera frame start `0`;
- TAA/jitter disabled and exposure fixed at `1.0`;
- a new renderer, scene, accumulation history, moments history, and
  reservoir history for every repeat;
- one cached `wgpu::Device` per test process, avoiding repeated
  headless Metal device teardown without sharing renderer resources.

Run the complete same-adapter stability gate and both negative controls with
one device kept alive for the entire qualification:

```shell
cd native/shared
cargo test --release --test golden_render \
  qualify_pt_oracle_hardware -- --ignored --exact --nocapture
```

`BLOOM_REQUIRE_RAY_QUERY=1` converts an unsupported/incorrectly
packaged adapter into a test failure. Without it, a genuine lack of a
non-CPU ray-query adapter emits a structured JSON skip. A ray-query
adapter that fails device creation always fails; it never silently
skips. Each repeat uses fresh renderer state while the device remains
alive, and the default repeat count remains one so ordinary test cost
does not increase.

Expected cross-backend differences are limited to low-amplitude
floating-point intersection, interpolation, texture-sampling, and
denoiser rounding near edges. Black regions, block trails, non-finite
values, uninitialized history, broad energy shifts, and coherent
motion trails are never backend variance. Do not widen tolerances to
accept them.

On failure, `native/shared/target/golden-artifacts/<test>/` contains
`expected.png`, `actual.png`, `absolute-diff.png`, `heatmap.png`, and
`metrics.json`. The JSON records the commit, OS/architecture, adapter,
backend, driver, supported/enabled features, seed, sequence starts,
repeat index, frames/spp, measured render wall time, fault control, and all
comparison metrics.
Set `BLOOM_GOLDEN_DIAGNOSTICS=1` for named captures:

- progressive: accumulated output, pipeline write probe, depth,
  normal, albedo, sun visibility, primary-ray agreement, raw radiance;
- realtime: denoised output, motion, raw radiance, history length,
  variance.

The query-heavy primary-ray probes (debug views 6–19) are
source-stripped from normal production shaders. They create ten extra
inline ray-query objects per thread and are compiled only when those
views or golden diagnostics are explicitly requested. This is
pixel-neutral for production while reducing Metal kernel state.

#### 2026-07-22/23 Metal root-cause audit and correction

On the audited dirty worktree based on commit `e498433`, macOS 26.5 and
an Apple M1 Max Metal ray-query adapter reproduced the issue. One
recorded progressive run had mean RGBA error `23.404968262`, mean RGB
error `31.206624349`, `58.0291748%` outlier pixels, maximum error `204`,
and SSIM `0.758563206`. The realtime test showed the same failure class
as localized dark block trails under motion.

A `git archive` of clean `e498433` was then run against the same adapter.
Its progressive output reproduced the historical PNG byte-for-byte, but
required `167.10 s` for 300 frames. Clean-main realtime still had not
completed after five minutes and was terminated to avoid further GPU
stress, so no clean-main realtime visual pass is claimed. This classifies
the original progressive baseline as visually stable but the Metal query
path as pathologically slow; the broad block corruption depended on the
newer renderer state in the audited worktree rather than on clean main
alone. The backend loop was nevertheless the first bad stage in that
state and a severe performance defect on both clean and dirty trees.

Named captures located the first divergent stage before accumulation:
the pipeline write, depth, normals, and albedo were clean, while sun
visibility was almost entirely black and raw radiance already contained
workgroup-shaped holes. The cause was not stochastic tolerance, history,
resource lifetime, or the diagnostic-query count. Naga 29.0.1 lowers a
Metal ray query differently from DX12/Vulkan:

1. `rayQueryInitialize` emits the complete synchronous
   `intersector.intersect(...)` and sets `ready = true`.
2. On the non-modern Metal path, `rayQueryProceed` only reads `ready`; it
   neither advances the query nor clears the flag.
3. The canonical WGSL `while (rayQueryProceed(...))` loop therefore does
   not terminate, even though the committed intersection is already ready.

All PT, HW-SSGI, and HW-WSRC query loops are now backend-specialized.
Metal compiles a constant-false proceed branch and reads the committed
intersection immediately; DX12/Vulkan compile the original proceed loop.
Generated Metal 2.4 source contains the synchronous intersection behind a
constant-false guard, while generated HLSL shader model 6.0 retains
`RayQuery.Proceed()`. The same WGSL sources validate through Naga for both
variants. This is a strict Metal correctness and performance improvement
and leaves non-Metal query semantics unchanged.

The first corrected visibility probe exposed a second, independent error:
PT negated the legacy primary-light vector even though the raster shader
and public setter consume it as the vector from the shading point toward
the sun. The PT upload and golden scene now use that convention directly,
matching the documented CPU command (`0.5 1.0 0.3`). The old checked-in
goldens were flat, sunless images; the corrected baselines are brighter and
contain coherent cast shadows whose direction and silhouettes agree with
the 256-spp CPU reference. They were updated only after the raw-radiance,
visibility, accumulated-output, and CPU-reference evidence was inspected.

Source-stripping debug views 6–19 remains worthwhile: the normal production
kernel has two ray-query objects instead of twelve. It reduces Metal kernel
state and compile cost, but the audit no longer presents it as the root-cause
fix.

On 2026-07-23 the corrected unified Apple M1 Max / Metal qualification passed all
three progressive and all three realtime repeats with byte-identical output
(`mean/max/outliers = 0`, `SSIM = 1.0`). Across two qualification runs under
different concurrent system load, measured render wall times (including
command submission and the final readback) were `851–3537 ms` for 300
progressive frames and `187–370 ms` for 48 realtime frames; the complete
six-run plus two-fault qualification took `7.58–23.16 s`. The previous invalid-loop path was
dramatically slower and became visually corrupt with the audited renderer
changes. These wall times are recorded in
failure JSON and pass logs as a local regression signal; issue #128 owns
uncapped per-pass GPU timestamp baselines. A DX12/Vulkan hardware result is
still required before cross-backend qualification is complete.

The negative controls must fail before a hardware result is accepted:

```shell
cd native/shared
BLOOM_REQUIRE_RAY_QUERY=1 BLOOM_PT_TEST_FAULT=brdf-energy \
  cargo test --release golden_pt_progressive -- --nocapture
BLOOM_REQUIRE_RAY_QUERY=1 BLOOM_PT_TEST_FAULT=reprojection \
  cargo test --release golden_pt_realtime_motion -- --nocapture
```

The first scales path radiance by 25%. The second shifts temporal
reprojection, deliberately bypasses the depth guard, and trusts the
misaddressed history over fresh radiance—the catastrophic cross-surface
history class the motion oracle must reject. Production uses zero offset,
keeps validation enabled, and retains the current sample; those constants
compile away. A successful negative-control test command is therefore an
error: each command above is expected to exit non-zero with golden
artifacts. The unified qualification verifies both failures automatically.

### CPU sanity oracle

`bloom-reference` has a procedural `pt-golden` scene with the exact
floor/cube geometry, materials, camera, sun, and deterministic seed:

```shell
cd tools/bloom-reference
cargo run --release -- \
  --builtin pt-golden \
  --out ../../native/shared/target/golden-artifacts/pt-reference.png \
  --metadata ../../native/shared/target/golden-artifacts/pt-reference.json \
  --width 256 --height 256 --spp 256 --bounces 8 --seed 0 \
  --camera 5 4 7 0 0.5 0 50 \
  --sun-dir 0.5 1 0.3 --sun-intensity 1.2
```

This is an energy/occlusion sanity check, not a whole-frame numeric
golden. Three intentional model/display differences dominate raw
RMSE: the CPU tracer renders the analytic environment on primary
misses while the GPU PT preserves the raster clear/sky; the CPU tracer
uses environment NEE/MIS while the GPU samples its sky on bounce
misses; and the CPU result uses its fixed ACES+sRGB output rather than
the engine HDR/post pipeline. Geometry silhouettes, direct shadows,
occlusion, material ordering, and the direction of indirect energy
must still agree.

Metal and DX12/Vulkan hardware runners should execute the stability
gate, both negative controls, and upload `target/golden-artifacts` on
failure. Hosted jobs without a ray-query device must not be presented
as coverage; their structured skip is only an applicability result.
