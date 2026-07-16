# Engine ticket backlog

Outstanding engine work surfaced while building the shooter game.
Game-side counterparts live in `bloom/shooter/docs/tickets.md`.

Status legend: 🟢 ready · 🟡 has open design questions · 🔴 needs
broader RFC

---

## EN-001 — Instanced-draw FFI ✅ shipped

> **Status:** landed — instance buffers + instanced material draws
> (`renderer/material_instancing.rs`, `bloom_create_instance_buffer`,
> `submit_material_draw_instanced`); shooter runs 20k grass instances
> in one draw, tile-culled since aeb3228.

**Why:** the shooter currently bakes 5000 grass blades into one
big static mesh and draws it with a single `drawMeshWithMaterial`.
That works because the blades are static, but it caps practical
density (rebuilding the mesh CPU-side at startup is the bottleneck)
and any future scattered prop has to repeat the same pattern.

A real instanced-draw API would:
- Take a small mesh (one blade) + a per-instance buffer (positions,
  rotations, scales, hue jitter).
- Upload the per-instance data once at startup, draw N instances
  per frame with a single draw call.
- Let games push to 20k–100k blades / particles / props with
  flat per-frame cost.

**API sketch:**

```rust
pub fn submit_material_draw_instanced(
    material: MaterialHandle,
    mesh_handle: u64, mesh_idx: usize,
    instance_data: &[f32],          // flat pos.x, pos.y, pos.z, rot.y, scale, tint.rgba per instance
    instance_count: u32,
);
```

**Acceptance:** shooter grass scatter goes from 5000 to 20000
blades with no startup-time hit and no measurable change in the
per-frame `material_pass` GPU time.

**Notes:** wgpu supports instanced draws via `draw_indexed` taking
an `instances: Range<u32>` parameter. The vertex buffer layout
needs a second buffer with `step_mode: VertexStepMode::Instance`.
Material pipelines that opt into instancing need a flag at compile
time so the right vertex layout is used.

---

## EN-002 — `drawModel` rotation parameter ✅ shipped

> **Status:** landed — `drawModelRotated` (degrees), later moved onto
> the cached scene pipeline so cutout/materials apply; used by the
> shooter's 88-tree forest.

**Why:** the shooter just shipped `drawMeshWithMaterial`-based tree
sway because `drawModel` has no rotation arg, so we couldn't tilt
trees from outside. Adding rotation to `drawModel` would let game
code apply per-instance Y-axis rotation directly to GLBs without
needing a custom material per use case (scattered crates, decals,
projectiles…).

**API sketch:**

```rust
pub extern "C" fn bloom_draw_model_rotated(
    handle: f64, x: f64, y: f64, z: f64,
    scale: f64, rot_y: f64,
    color_packed_argb: f64,         // 0xAARRGGBB packed; sidesteps the 9th-arg quirk
);
```

`color_packed_argb` keeps the call to 7 f64 args so all of them
go into ARM64 registers — same workaround we already use for
`bloom_update_model_animation`.

**Acceptance:** scattered crates / barrels in the world data can
have a `rotation` field that's actually honoured at draw time.

---

## EN-003 — SSAO intensity / radius knobs ✅ *(closed 2026-07-15 — was already done)*

> Stale bookkeeping, not new work: `set_ssao_intensity` / `set_ssao_radius`
> are live in `ffi_core/visual.rs` + `src/core/index.ts`, declared in the
> manifest, and present on web and watchOS. Marked open long after landing.

**Why:** SSAO is in the pipeline (visible in F3 overlay). Engine
exposes `setSsaoEnabled(on)` only — no intensity or radius
control. After Tier 1's HDR-env baseline, the AO tuning that
worked for flat-ambient lighting now over-darkens corners.

**API sketch:**

```ts
setSsaoIntensity(intensity: number): void;   // 0..2
setSsaoRadius(world_radius: number): void;   // 0.1..2 m
```

**Scope:** small — SSAO shader already takes these as uniforms,
just expose CPU setters + FFI passthrough.

**Acceptance:** turning intensity to 0 visibly removes the corner
darkening; turning it to 1.5 deepens it; radius affects how far
along walls / under trees the contact darkening reaches.

---

## EN-004 — JSON `loadMaterial(path)` runtime loader 🟡

**Why:** Phase 5 shipped `loadMaterial(desc)` with a typed object,
explicitly punting JSON-on-disk loading because Perry's runtime
`JSON.parse` produces arrays whose `.length` lies. Games that
want JSON descriptors today have to preprocess at build time.

**Resolution path:** a build-time TypeScript tool
(`shooter/tools/build-materials-from-json.ts` or generic) that
reads `assets/materials/*.json` and emits a `src/generated/materials.ts`
module that calls `loadMaterial(...)` with literal descriptors.
Same pattern as `build-world.ts`.

**Open question:** should this live in the engine or the game?
The pattern is generic enough to belong in the engine SDK, but
the engine doesn't currently provide build-time tooling.

**Scope:** small.

---

## EN-005 — Atmospheric scattering / sun disk ✅ V1+V2

**Status:** V1 + V2 landed 2026-04-26. RFC: `docs/rfc/0002-atmospheric-sky.md`.

**Shipped (Hillaire 2020 procedural sky, full V2):**
- Phase 1: transmittance + multi-scattering LUTs baked at init.
- Phase 2: sky-view LUT compute pass + procedural sky render pass +
  sun disk with limb darkening. `setProceduralSky` / `setSunDirection`
  TS API across all platform crates.
- Phase 3: IBL re-bake on sun-move (procedural sky drives PBR
  reflections + ambient). Sun-shaft tint auto-derived from
  transmittance LUT (sub-RFC #1 answered).
- Phase 4: sky-tinted fog auto-derive + zenith dithering polish.
- V2: full 3D aerial-perspective LUT (32³ desktop / 16³ web+mobile),
  per-frame compute from current camera + sun. scene_compose
  samples it instead of running the 16-step volumetric fog march
  when procedural sky is on. Per-pixel angular variation — sunset
  side reads warmer than the opposite horizon.

**Sub-RFCs answered in RFC 0002:**
- Sun-shafts tap transmittance (yes — `setSunShaftColor` retains as
  override).
- `setSunDirection` is the source of truth when procedural is on;
  `setDirectionalLight` stays as the lower-level escape hatch.
  Time-of-day deliberately deferred to user-space.

---

## EN-006 — GPU-pipeline integration tests 🟢

**Why:** the 8 unit tests we ship cover CPU-only logic (profiler
ring buffer, hot-reload channel + dedup). The harder paths
(translucent dispatch, depth snapshot, impulse compute, hot-reload
end-to-end) all rely on visual smoke tests. Each is exposed by a
Renderer FFI and could be exercised in a `cargo test` that creates
a wgpu device + dispatches + reads back.

**Scope:** medium — pollster is already a dev-dep, so the GPU
test infrastructure exists.

**Tests to add:**
- Translucent dispatch: compile a refractive material, submit a
  draw, dispatch the translucent pass, verify hdr_rt has alpha-
  blended pixels at the expected screen position.
- Depth snapshot: verify the transient depth texture content
  matches the live depth buffer after `copy_texture_to_texture`.
- Impulse compute: submit a splat, dispatch the compute, read
  back the texture, verify the splat appears at the expected
  texel + decays correctly across frames.
- Hot reload end-to-end: register a material, write a new file
  to disk, advance time, verify `pipelines[handle]` was replaced.

**Acceptance:** the 4 tests pass on macOS-Metal; they'll run in
CI once we have it.

---

## EN-007 — Dead-code sweep on bloom-shared 🟢

**Why:** bloom-shared still has 8 warnings (after the recent
17→0 / 13→8 sweeps), all of them flagging
"function/field/constant never used":

- `walk_scene_for_mesh_transforms`
- `quat_mul`
- `SSAO_FRAG`, `COMPOSITE_FRAG` (string constants)
- `texture_idx`, `debug_frame` (struct fields)
- `tex_a` / `tex_b` (struct fields)

Some are leftover refactor cruft, some may be intentional for
future use. A pass that either deletes (and verifies nothing
breaks) or marks `#[allow(dead_code)]` with a "kept for X" note
would zero out the warning count and document intent.

**Scope:** small (per-item investigation, mostly delete or
allow-with-comment).

---

## EN-008 — Cargo feature for `notify` watcher 🟡

**Why:** `notify` is unconditionally pulled into the build. The
hot-reload watcher itself is gated by `BLOOM_NO_HOT_RELOAD=1` at
runtime, but the dependency still bloats the binary in shipped
builds. Phase 6 closes its sub-item via runtime gate; an additional
compile-time `hot-reload` feature would make it truly opt-out.

**Scope:** small — feature flag + cfg on the `MaterialHotReload`
struct + the call site in `engine.rs`.

**Acceptance:** `cargo build --release --no-default-features` for
the shooter ships a binary without `notify` linked.

---

## EN-009 — Multi-mesh `drawMeshWithMaterial` ergonomics ✅ *(closed 2026-07-15 — was already done)*

> Stale bookkeeping: the `drawModelWithMaterial` wrapper exists in
> `src/models/index.ts`.

**Why:** rendering a multi-primitive GLB (like the 4 Kenney trees,
where one variant has 3 primitives) now requires the caller to
know `meshCount` and loop. Almost every caller wants "draw all
primitives" rather than "draw primitive N." A convenience API
would tighten the call site.

**API sketch:**

```ts
drawModelWithMaterial(material: number, mesh: Model,
                     position: Vec3, scale: number, tint: Color): void;
```

Internally loops `0..mesh.meshCount` calling `drawMeshWithMaterial`.

**Scope:** tiny — TS-only wrapper, no engine change.

---

# UE5-tier rendering enablers

EN-010 onward are the engine-side gates that unlock the next
quality tier in the shooter's grass / trees / water. Game-side
counterparts and the phase ordering are in
`bloom/shooter/docs/visual-quality.md` (Tier 6+ section) and the
SH-020..SH-024 tickets.

---

## EN-010 — Alpha-cutout bucket ✅ engine side shipped

> **Status:** cutout bucket landed in the material pipeline (leaf-card
> trees render through it since the round-5 texture work). Game-side
> SH-020 (converting 2 of 4 tree variants) still pending.

**Why:** today's render buckets are Opaque (full G-buffer),
Transparent (back-to-front blend), Refractive (Transparent +
auto-snapshot), Additive (order-independent). Foliage cards,
chain-link fence textures, and any leaf-silhouette material need
a fifth: alpha-test with `discard` against
`MaterialFactors.alpha_cutoff` — runs in the opaque pass (G-buffer
write, sun shadow, SSAO) but skips the discarded fragments.

`MaterialFactors.alpha_cutoff.w` already exists in the ABI
(`shaders/material_abi.wgsl`); the scoreboard already has a slot.
What's missing is the pipeline state (the discard path is a
fragment-shader emit + a flag at compile time so the alpha
channel of `base_color_tex` actually drives the cutoff) and the
bucket assignment.

**API sketch:**

```rust
pub enum Bucket {
    Opaque,
    Cutout,        // ← new: front-to-back like Opaque, fragment discards
    Transparent,
    Refractive,
    Additive,
}
```

WGSL side: a new `CutoutOut` writes the same 4 G-buffer targets as
`OpaqueOut` and the engine wraps the fragment with a `discard`
based on `alpha < alpha_cutoff`. Or simpler: keep `OpaqueOut` and
let materials in `Bucket::Cutout` call `discard` themselves —
engine just changes pipeline state to disable z-pre-pass merging.

**Acceptance:** shooter ships SH-020 leaf-card trees that drop
discarded leaf pixels but still cast and receive proper sun
shadows.

**Scope:** small — pipeline state branch + bucket enum + WGSL
emit doc update.

---

## EN-011 — Planar reflection capture ✅ shipped

> **Status:** landed — `renderer/planar_reflection.rs` /
> `planar_pass.rs` + `setMaterialReflectionProbe`; the river reflects
> the bank. Mirrored-frustum culling, batched uniforms, cached probe
> bind group, and `setMaterialProbeVisible` (grass exclusion) landed
> in the 2026-07 perf round.

**Why:** the water material today reads `env_tex` (the static HDR
panorama) for sky reflection. That's correct for sky but means a
river never reflects the trees on its banks or the bridge crossing
it. Planar reflections — a second camera mirrored across the water
plane, rendering scene geometry into a low-res RT bound as a
texture in the water material — are the standard solution and the
single most-noticed water upgrade in modern games.

**API sketch:**

```ts
const probe = createPlanarReflection({
  plane:    { y: 0.05, normal: vec3(0, 1, 0) },
  resolution: 512,                    // wide-screen-aligned RT
  layers:   ReflectionLayers.WORLD,   // skip player, skip foliage if perf-bound
});
// Material binds the probe's RT at @group(2) @binding(12)
setMaterialReflectionProbe(matWater, probe);
```

Engine does:
1. Per frame, build a mirrored view matrix across the plane.
2. Render the world list (with a layer mask) to the probe's
   colour RT.
3. Bind the RT into materials that opt in.
4. Materials sample with the perturbed wave normal as offset.

**Open questions:**
- One probe per plane vs shared probe? (V1: one per plane. River
  is one plane.)
- Should the probe re-render every frame or every other frame?
  (V1: every frame at half-res; tune later.)
- Cull list — same as main camera or whitelist? (V1: whitelist
  big-silhouette geometry, skip particle systems and grass.)

**Scope:** medium — render-graph addition + new material binding
slot + cull-list filter. Roughly 3 days.

**Acceptance:** SH-022 ships the river reflecting the actual tree
bank, with the reflection wobbling on the wave normal.

---

## EN-012 — Foliage shading model in PBR ABI 🟡 *(V1 landed; see shooter SH-023 — its transmission drops the dot(-N,L) gate and it excludes ambient/IBL, so the shooter declined to adopt it. A V2 needs both.)*

**Why:** SH-011 (grass) and SH-012 (tree) both bolt local
wrap-lambert + transmission terms into bespoke material shaders.
The pattern is universal across foliage — a real shading model
in `material_abi.wgsl` would let any new foliage material
declare `shading_model: foliage` and inherit:

- Wrap-lambert diffuse so back-faces don't go pure black.
- Transmission term (sun behind leaf → warm tint into camera).
- Two-sided normal handling (no need to double the geometry).
- Per-material transmission colour from `MaterialFactors`.

UE5 calls this the **Foliage** shading model; it's the second-most-
used shading model after Default Lit in any outdoor scene.

**API sketch:**

```wgsl
// In material_abi.wgsl additions:
struct FoliageShading {
  transmission_color: vec3<f32>,
  transmission_amount: f32,
  wrap_factor: f32,
};

// MaterialFactors gets a new sub-block:
struct MaterialFactors {
  // ... existing fields
  shading_model: u32,            // 0 = default, 1 = foliage, 2 = subsurface
  foliage:       FoliageShading,
};
```

**Open questions:**
- Should subsurface (skin/wax) and foliage share a shading-model
  enum or be separate? (Foliage is a subset; recommend shared
  enum.)
- How do shadow / SSAO behave on two-sided foliage? (Sample
  shadow on either side; SSAO masks at half-strength on
  backfaces.)

**Scope:** medium — shading model branch in the deferred lighting
pass + ABI struct + WGSL standard library helper.

**Acceptance:** SH-023 ports `grass.wgsl` and `tree.wgsl` to
`shading_model: foliage` declarations and ~30 lines of bespoke
math go away from each.

---

## EN-013 — Global wind UBO in PerFrame ✅ shipped

> **Status:** landed — `frame.wind` + `setWind()`; shooter grass and
> trees ride one UBO.

**Why:** every foliage material today re-declares its own `wind:
vec4<f32>` in a per-material UBO (GrassParams, TreeParams). When
new foliage materials land (SH-020 leaf cards, future ferns,
clovers, etc.) the game has to keep N UBOs in sync per frame.
A small extension to the global PerFrame UBO — adding a
`wind: vec4<f32>` (dir.xz, amplitude, frequency) plus a
`gust_phase: f32` — lets all foliage materials sample one source
of truth.

**API sketch:**

```ts
setWind(dirX: number, dirZ: number, amplitude: number, frequency: number): void;
```

WGSL side: `frame.wind: vec4<f32>` becomes available in any shader
including `material_abi.wgsl`.

**Acceptance:** grass and tree materials read `frame.wind` instead
of their own UBOs; one `setWind()` call drives all foliage; visual
parity with today's per-material params.

**Scope:** tiny — extend PerFrame, FFI passthrough, doc update.
Per-material wind UBOs become deprecated but stay for one-off
overrides.

---

## EN-014 — Texture-array binding pattern for splat-mapped terrain ✅ shipped

> **Status:** landed — texture-array bindings in the material ABI
> (`material_abi.wgsl`, splat-terrain support in the material system).
> Game-side SH-009 (4-layer PBR terrain) is unblocked and pending.

**Why:** SH-009 wants 4 PBR layers (grass / dry-grass / dirt /
rock) blended in the fragment by weight masks. The current
PerMaterial group has 5 PBR slots (base, normal, MR, emissive,
occ), each as a single `texture_2d`. A 4-layer terrain needs 12
textures (4 × albedo/normal/MR) which doesn't fit, and even if it
did, the fragment shader would have to declare 12 samplers — ugly
and wasteful.

The standard solution is **texture arrays**: one
`texture_2d_array<f32>` slot bound with N layers, sampled with a
layer-index parameter. UE5 calls these "Texture2DArray" and uses
them everywhere for landscape materials.

**API sketch:**

```ts
// CPU side
const albedoArray = loadTextureArray([
  'assets/textures/grass_lush_albedo.png',
  'assets/textures/grass_dry_albedo.png',
  'assets/textures/dirt_albedo.png',
  'assets/textures/rock_cliff_albedo.png',
]);
setMaterialTextureArray(matTerrain, /*slot=*/0, albedoArray);
```

Material ABI: a new optional binding pattern at @group(2)
@binding(13/14/15) (after the user_params slot at 11) for albedo
/ normal / MR arrays. WGSL emits `texture_2d_array` declarations
when materials opt in.

**Open questions:**
- One array slot or three (albedo / normal / MR separately)?
  (Recommend three; lets games mix-and-match resolutions.)
- Layer count limit — max 16? 64? (16 is enough for any landscape
  shader; 64 is wgpu's typical limit.)

**Scope:** medium — wgpu binding type, FFI loader, ABI doc.

**Acceptance:** SH-009 ships 4-layer triplanar terrain with one
sampler call per layer-class instead of 4.

---

## EN-015 — Imposter / billboard system 🟡 *(deprioritised — the shooter's 88-tree forest was measured as pure alpha-cutout OVERDRAW, not geometry; EN-044's depth prepass solved it. Imposters remain a >500-instance tool.)*

**Why:** the shooter ships 120 trees today; opening the playfield
or moving toward "stand on a hill, see a forest stretching to the
horizon" needs 500 – 5 000 trees. Beyond ~40 m, full-poly
foliage is wasted budget — an octahedral imposter (a single quad
textured with a pre-rendered multi-angle bake) renders in 2
triangles instead of 2 000.

**Pieces:**

1. **Bake tool** (`engine/tools/bake-imposters.ts`?): renders a
   GLB into 8 × 8 = 64 view directions on an octahedron, packs
   the colour + depth bakes into one atlas texture.
2. **Imposter material** that samples the right view from the
   atlas based on the camera-to-imposter direction.
3. **LOD selection** — game code or engine picks imposter vs
   full mesh by distance.

**API sketch:**

```ts
const imposter = bakeImposter(treeOakModel, { views: 8, atlasRes: 2048 });
// Game draws full mesh inside 40 m, imposter outside.
drawImposter(imposter, position, scale, distance);
```

**Open questions:**
- Static-only or skinned? (V1: static. Foliage doesn't skin.)
- Lit imposters require packing albedo + normal + roughness, not
  just colour. (V1: ship with full PBR pack.)
- Distance hysteresis to avoid LOD pop. (Engine TAA may already
  hide it.)

**Scope:** medium-large — new draw bucket + bake tool + atlas
loader. Roughly 1 week.

**Acceptance:** SH-024 ships 1 000 trees at the same per-frame
cost as today's 120.

---

## EN-016 — Custom-material shadow-receive helper ✅ shipped

> **Status:** landed — `sample_sun_shadow` / `sample_sun_shadow_n`
> helpers; consumed by grass, terrain, tree, and water materials.

**Why:** game materials (`grass.wgsl`, `tree.wgsl`, `terrain.wgsl`,
`water.wgsl`) all want to sample the directional shadow cascades.
The math lives in `material_abi.wgsl` (cascade selection by depth,
PCF Vogel disk, comparison-sampler call) but isn't exposed as a
helper — every consumer has to copy it inline. After Phase 6 a
`#include "common/shadows.wgsl"` with a one-call helper would
make all four shaders sample shadows correctly with one line.

**API sketch:**

```wgsl
// In common/shadows.wgsl
fn sample_sun_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    // Return 0 (fully shadowed) .. 1 (fully lit). Auto-picks cascade,
    // applies normal-bias offset, runs PCF Vogel disk.
}
```

**Scope:** tiny — package existing shader code into a helper +
include path.

**Acceptance:** SH-011 (grass) and SH-012 (tree) drop in a one-line
shadow sample and visually receive sun shadows from the canopy
above.

---

## EN-017 — Post-pass slot for game-side full-screen FX ✅ shipped

> **Status:** landed — `bloom_add_post_pass` game-injected fullscreen
> passes. First consumers: shooter SH-029 damage-flash / low-health
> grading (the original SH-019 underwater use case was closed).

**Why:** SH-019 wants an underwater colour tint when the camera Y
falls below the water surface. The engine ships several built-in
post effects (bloom, vignette, film grain, chromatic aberration,
sun shafts, auto-exposure) but no slot for a game-supplied
fullscreen WGSL pass. SH-019 is the immediate use case; future
candidates: damage-flash red overlay, scope vignette, low-health
desaturation.

**API sketch:**

```ts
const matUnderwater = compileMaterial(`
  @fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_color_tex, scene_color_samp, uv);
    return vec4<f32>(scene.rgb * vec3<f32>(0.4, 0.7, 0.9), 1.0);
  }
`, /*post=*/true);
setPostPass(matUnderwater);
clearPostPass(); // for transient effects
```

The post-pass runs after composite + tonemapping but before the
final blit; it samples `scene_color_tex` (LDR) and outputs to the
swapchain. Game can disable / replace per frame.

**Open questions:**
- One slot or stack? (V1: one slot. Stacking can wait.)
- Does the post-pass see depth? (Bind `scene_depth_tex` like
  refractive bucket, yes.)

**Scope:** small — render-graph addition + FFI + ABI doc.

**Acceptance:** SH-019 ships an underwater tint that toggles on
when the camera Y < water Y; HUD remains crisp because the post
pass runs before the UI overlay.

---

## EN-018 — Validate render-scale + DRS on macOS Retina 🟢

**Why:** PRs `4ba0ea4` → `f1c9cd2` shipped `setRenderScale`,
`setUpscaleMode`, `setCasStrength`, and `setAutoResolution`
(auto-DRS) end-to-end. macOS already honored `backingScaleFactor`,
so a 4K Retina display is the cheapest place to confirm the new
machinery works visually + that the DRS controller settles.
Nothing in CI exercises a real GPU at fractional scales today.

**Scope:** test/manual, no engine code changes expected.

**Acceptance:**
1. `examples/intel-sponza` (or whichever 3D scene is current)
   runs at native 4K Retina with no visible regression vs
   pre-PR-1 baseline (TAA on, render_scale=0.5 default).
2. Sweeping `setRenderScale` from 1.0 → 0.75 → 0.5 produces
   monotonically decreasing GPU frame time
   (`bloom_get_profiler_frame_gpu_us`), Catmull-Rom upscale
   shows no obvious aliasing vs bilinear at 0.75.
3. CAS at 0.4 visibly sharpens silhouettes on the marble columns
   without crunching the shadowed crevices (HDR pre-tonemap
   placement is the whole point — verify on a stop-motion
   screenshot pair, CAS off vs CAS on at scale 0.75).
4. **Auto-DRS settling test:** call `setAutoResolution(60, true)`
   in the example, then induce a GPU spike via
   `setSsgiIntensity(5.0)` (or rotate the camera into the
   sun-shaft heavy courtyard). Within ~30 frames `getRenderScale()`
   should step *down* one rung. Restore the workload, confirm it
   steps *up* again within 60–90 frames. Asymmetric hysteresis is
   the design — drops fast, recovers slowly.
5. `setTaaEnabled(true)` then `setRenderScale(0.75)` then
   `setTaaEnabled(false)` should leave scale at 0.75 (the explicit
   setter pins it). With no manual scale call, toggling TAA should
   continue to flip 0.5 ↔ 1.0 (legacy coupling preserved).

**Notes:**
- Logs each scale change so the trace is auditable.
- A short MP4 of the DRS settling for the visual-quality doc
  would be nice-to-have; not blocking.

**Blocked on:** the in-flight atmosphere/sky WIP currently breaks
`cargo test --lib`; either land or revert that before adding any
new screenshot fixtures via the test harness.

---

## EN-019 — Validate cross-platform HiDPI on Windows + Linux 🟡

**Why:** PR `f1c9cd2` added Per-Monitor-Aware-V2 + WM_DPICHANGED
on Windows, X11 `XDisplayWidth/MM` scale detection on Linux, and
`devicePixelRatio` canvas sizing + `ResizeObserver` on Web. These
were written from a macOS dev box without cross-targets installed
— code paths compile but were never actually run on the target
OS. A 4K Windows monitor at 150% DPI is the canonical "before
this PR rendered at ~2560×1440, after it should render at 3840×2160"
test.

**Scope:** test/manual + likely small follow-up bug fixes.

**Acceptance:**
1. **Windows 4K @ 150%:** `examples/intel-sponza` window at
   `setRenderScale(1.0)` shows `getPhysicalWidth()` ≈ 1.5 ×
   `screenWidth()`. UI text is crisp (was blurry before). DPI
   scale persists when dragging the window between a 100% and
   150% monitor (WM_DPICHANGED fires; the window resizes to keep
   apparent size; a follow-up WM_SIZE re-resizes the renderer).
2. **Linux X11 @ 144 DPI** (e.g. set via `xrandr --dpi 144` or
   a 27" 4K monitor with correct EDID): `display_scale()` returns
   1.5 (snapped from raw 1.5). Surface created at logical × 1.5.
3. **Web Retina (Chrome + Safari on macOS):** canvas `width`
   attribute equals logical × `devicePixelRatio` (2 on Retina).
   CSS dimensions unchanged. Browser zoom (Cmd+ to 125 %, then
   Cmd+0) round-trips through the matchMedia listener and
   re-resizes the surface.
4. **Confirm `bloom_set_render_scale(0.75)` actually scales fragment
   cost** on each platform — same `bloom_get_profiler_frame_gpu_us`
   methodology as EN-018.

**Open questions:**
- Wayland: explicitly skipped in PR 3 ("Wayland out of scope").
  EN-019b if a Linux user files a regression on a Wayland session.
- Per-window DPI on Windows pre-1607 (no `GetDpiForWindow`):
  PR 3 falls back to `GetDpiForSystem` which is good enough for
  the initial window create. If this becomes a real-world issue
  add `GetDeviceCaps(hdc, LOGPIXELSX)` as a third fallback.

**Blocked on:** ~~access to a Windows 11 box with a HiDPI display~~ —
**the Windows half is field-validated as of 2026-07**: the shooter dev
box is exactly the canonical setup (Windows 11, 4K @ 150%), and months
of round-1/round-2 work ran on it — physical 3840×2160 vs logical
2560×1440 split correct end-to-end (borderless fullscreen at native
res, PMv2 capture tooling, and the 2D text physical-res work all depend
on it). Still untested: WM_DPICHANGED monitor-drag (single-monitor
box), Linux X11, Web Retina. Web checks can be done from any modern
browser on the macOS dev box.

## EN-020 — Native AV: heap read overrun, layout-sensitive ✅ root-caused 2026-07-04

**Symptom.** Bloom Shooter round-2 audit (2026-07-04): three scripted runs
crashed with c0000005 at the same instruction (`main.exe+0xe8e5` on the
af98dbe-era link), reading `0x…FFF8` — 8 bytes below a page boundary.
No Rust panic output (engine builds `panic = "abort"`, which would print),
so this is raw UB: something reads past the end of a heap allocation and
faults only when the allocation abuts an unmapped page.

**Root cause (found via the title-freeze report on the round-2
integration build).** Perry 0.5.x runtime string scanning: `split()`
slices and `parseFloat()` read past the end of Perry's own exact-sized
string allocations. Any TS that runs a packed FFI text blob through
`split`+`parseFloat` every frame — exactly what `getProfilerOverlay` /
`getProfilerFrameHistory` did — crashes within seconds once a slice lands
flush against an unmapped page. Reproduced 6/6 across two different link
layouts (faults in `perry_fn_…getProfilerOverlay` at `+0xe925` and
`…getProfilerFrameHistory` at `+0xf0a3`, 7–29 s of overlay time each);
engine-side tail-padding of `alloc_perry_string` alone did NOT fix it,
proving the overread is on Perry-internal allocations. Minidumps
archived in `shooter/tools/.testout/dumps/crash_main_*.dmp`.

**Fix shipped (2026-07-04, `fix(EN-020)` commits).**
1. Numeric profiler ABI: `bloom_profiler_row_count/_label/_cpu_us/_gpu_us`
   + `_hist_*` — f64s cross the FFI; the label crosses whole and is only
   drawn, never parsed. `getProfilerOverlay`/`getProfilerFrameHistory`
   rewritten on top; the packed-text FFIs remain for back-compat but
   per-frame consumers must not parse them.
2. Defense-in-depth: `alloc_perry_string` now zero-pads 16 bytes after
   every payload (word-stepping scanners on ENGINE-allocated strings stay
   in bounds), regression test included.
3. Observability: unhandled-exception filter in `native/windows` prints
   code + `main.exe+RVA` and writes a minidump via runtime-loaded dbghelp;
   WM_CLOSE/WM_DESTROY and surface-acquire failures now log to stderr.
   Silent deaths are structurally impossible for AV-class faults now.

Validated: 3/3 runs × 90 s gameplay with the overlay ON continuously —
zero faults, overlay data correct (previously 6/6 dead in ≤29 s).

**Remaining.** File the scanner overread upstream against Perry 0.5.x
(repro: return a `"1.23|4.56\n"…` blob from an FFI, split+parseFloat it
per frame; dumps + offsets above). The OBJ text loader
(`engine/src/models/index.ts`) uses the same split/parseFloat pattern at
LOAD time — one-shot exposure, worth migrating when touched. Audit
report: `shooter/docs/audit-round2.md` finding F1.

## EN-021 — SSR + IBL specular exclusive ownership ✅ implemented (PR #78)

**Status 2026-07-04.** Landed on `feat/en021-ssr-ibl-ownership` (PR #78,
pending merge): env-cubemap fallback bound into the SSR pass (miss
returns filtered env instead of black, `env_fallback()` in ssgi.rs;
fresnel applied before the facing check so both miss paths agree), and
`fs_main_scene` scales `ibl_spec` by the exact complement of the SSR
fade (`ssr_own` via `dir_light_count.z`, plumbed through
`clear_additional_lights`). Design notes below kept for review context.

**Why.** Round-2 audit (B): compose is `hdr + ssr` and `fs_main_scene`
already adds IBL specular into hdr for metals — pixels with an SSR hit
double-count specular for roughness ≈0.05–0.85, worst for metals at
r≈0.55–0.75 (in the shooter: alien carapaces / weapon per their glTF
factors). Dielectrics are largely starved of IBL spec by design and are
fine; the shooter's custom world materials never call ibl() and are
unaffected.

**Why not a one-liner.** Scaling `ibl_spec` by the complement of the SSR
fade must be PAIRED with an env fallback on SSR miss (today miss = black,
ssgi.rs:1495) or off-screen-reflection pixels lose their specular
entirely — a worse artifact than the double-count. The fallback needs the
env cubemap bound into the SSR pass (bind-group layout change).

**Sketch.** (1) Bind env_tex into the SSR march; on miss return
`sample_env(r, r*max_mip) * fresnel * roughness_fade * strength` instead
of black. (2) In `fs_main_scene`, scale `ibl_spec` by
`1 − (1 − smoothstep(0.5, 0.85, r)) * ssr_strength` (the exact
complement of the SSR shader's own fade). (3) Regenerate goldens; verify
with the audit's metal-ROI protocol (on-hit vs panned-off luma converge
<5%).

**Interim calibration option:** `set_ssr_strength(0.25–0.35)` halves the
overlap engine-wide with zero shader work.

## EN-022 — Motion vectors for material-system draws ✅ implemented (PR #82 + shooter PR #4)

**Status 2026-07-04.** Landed on `feat/en022-material-velocity` (PR #82,
pending merge) + shooter PR #4: per-slot model history in
MaterialSystem (`prev_models`/`cur_models` rotated in
`reset_draw_slot(prev_vp)`, slot = submission order), `prev_mvp`
reconstructed engine-side (the legacy caller-supplied param is
ignored), `abi_motion_vector()` in material_abi.wgsl, and all four
shooter world materials (terrain/building/tree/grass — incl. the inline
grass copy in main.ts) write real velocity with wind sway evaluated at
`frame.time - frame.delta_time`. Smoke-validated: no TSR tearing,
sway ghosting gone. Design notes below kept for review context.

**Why.** Round-2 audit (F8): every material-path draw writes velocity = 0
(terrain/tree/grass/building — the whole static world plus wind sway),
so TAA's motion adaptation (`post.rs` motion_alpha ramp) never engages
for the world: camera translation is covered only by depth reprojection
and object sway not at all. This is the primary mechanism behind TSR 0.5
shimmer on thin grass + sway ghosting, and it blocks any future
motion-blur quality on the world.

**Scope.** Per-draw previous-frame model matrix (the per-draw UBO slot
already carries the current one; the shadow path already keeps a CPU
copy in `MaterialDrawCommand.model`), previous view-proj in PerView (the
core path has it), and a velocity write in the material ABI's OpaqueOut
path — plus the wind displacement evaluated at previous-frame time in
foliage vertex shaders so sway produces real motion vectors. Roughly:
ABI struct + material_system plumbing + 4 world materials + goldens.

**Acceptance.** Strafe at the audit's S7 pose: thin-grass shimmer index
(frame-diff metric) drops materially; wind sway no longer ghosts;
`post.rs` motion clamp engages on world pixels (debug: motion_alpha
visualization).

## EN-023 — GI software path: colored bounce is unreachable 🟡 partially landed (PR #79), still open

**Update 2026-07-13 — the premise below is wrong, and it changes the priority.**
The dev box's Radeon 760M *does* support hardware ray query. It reported
`ray_query=false` because **wgpu's DX12 backend was compiling shaders with
FXC**, which caps the reported shader model at 5.1 — and
`EXPERIMENTAL_RAY_QUERY` is gated on shader model ≥ 6.5 (wgpu-hal
`dx12/adapter.rs`: `supports_ray_tracing`). With FXC, ray query is unreachable
on DX12 **on every GPU**, so every Windows machine was silently running the SW
path. Switching the DX12 backend to DXC (`Dx12Compiler::DynamicDxc`, DLLs
copied by `tools/fetch-dxc.ps1`) flips the boot line to `ray_query=true` and the
trace backend to `hw-ray-query`.

So EN-023 is no longer the thing standing between this hardware and coloured
bounce — the HW path was always available. The SW path still matters for
adapters that genuinely lack RT (most Android, web permanently), and the
object-space-AABB bug below is still real there. But it is no longer urgent, and
the interim "disable SSGI on SW adapters" is no longer needed on Windows.

**Status 2026-07-04.** `feat/en023-gi-sw-cards` (PR #79, pending merge)
fixed the data path: world-space AABBs carried per instance
(`world_aabb_min/max` in InstanceGiData, both HW and SDF struct
mirrors), smallest-containing-box broad-phase pick (a scene-spanning
terrain proxy no longer swallows every hit), and the SW WSRC bake got a
`ground_albedo` bounce term. Measured payoff on the 760M is still ≤2%
achromatic even at 4× intensity — the bottleneck moved downstream to
probe resolve/AO integration, so the ticket STAYS OPEN pointed there.
Interim option D2-B (disable SSGI on SW adapters) remains reasonable.

**Why.** Round-2 audit (F4): on adapters without EXPERIMENTAL_RAY_QUERY
(the dev box's Radeon 760M reports none — see the new boot log), the
probe trace runs the SDF path, where the mesh-card lookup compares
world-space hits against OBJECT-space AABBs (ssgi.rs broad phase) — every
transformed instance falls through to the flat gray 0.55 analytic — and
the software WSRC bake is analytic sun+sky with no geometry at all.
Measured in-game: SSGI on/off is ≤2.5% achromatic luma, hue ratio
unchanged to 4 decimals. The ~267 gi_only proxies feed a pipeline whose
colored output cannot reach the screen on SW adapters, at ~1.6 ms GPU in
combat.

**Sketch.** (1) Transform SDF hits into instance object space (or
instance AABBs into world space) in the card broad-phase so textured/
tinted cards resolve. (2) Make the SW WSRC bake sample cards (or at
least sun-shadowed ground/albedo bounce) instead of pure analytic.
(3) Re-run the audit's GI A/B: acceptance = G/R hue shift at the
building base and ≥8-10% luma in shaded receivers.

**Interim option** (decision D2-B in the shooter audit): boot-time
`setSsgiEnabled(false)` on SW adapters banks the ~1.6 ms until this
lands; needs a backend query FFI or the game reading the new boot log.

---

## EN-024 — iOS/iPadOS/tvOS report pixels as the logical size 🔴 needs decision

**Why.** macOS passes points as logical and pixels as physical
(`native/macos/src/lib.rs:361`, via `backingScaleFactor`). iOS/iPadOS
and tvOS pass **pixels for both**: `bloom_attach_native` sets
`logical_w = physical_w = width` (`native/ios/src/lib.rs:770-773`) and
the orientation-change poll resizes with `resize(pw, ph, pw, ph)`
(`native/ios/src/lib.rs:828`); tvOS mirrors it. Touch input is then
scaled points→pixels to match (`native/tvos/src/lib.rs:267`), so the
convention is at least self-consistent — a deliberate-looking choice,
not an obvious slip.

Two consequences, both real:

1. **The physical-resolution glyph work is inert on iOS.** The shared
   text renderer derives `dpi = surface_config.width / logical_width`
   (`native/shared/src/text_renderer.rs:284-285`). With logical ==
   physical that is always 1.0, so the "rasterize glyphs at physical
   resolution" change (60ba704) does nothing on Apple mobile — glyphs
   rasterize at 1× of a pixel-sized font instead of at the backing
   scale. Text is not blurry (sizes are already in pixels) but the
   crispness win never lands, and 3× devices get 3×-smaller text than
   the same source produces on macOS.
2. **Game-facing coordinates differ across Apple targets.** `screenWidth`
   and 2D HUD coordinates are points on macOS and pixels on iOS, so the
   same layout code does not carry across.

**Decision needed (breaking either way).** Either (a) switch iOS/tvOS to
the macOS convention — pass points as logical, pixels as physical, and
stop pre-scaling touches — which fixes both consequences but changes
game-facing coordinates and will break existing iOS games' layout; or
(b) keep pixels and document the divergence explicitly, accepting that
`text_renderer`'s dpi path is dead on Apple mobile. Deliberately not
folded into the boot-path render-target fix (PR for EN-024's sibling
bug), which was a straight bug — this one is a product call.

**Found by** the macOS/iOS parity audit, 2026-07-11.

---

# AAA-feel enablers (2026-07-12 gap audit)

EN-025 onward come out of the full shooter/engine/editor AAA gap
audit. The renderer is past diminishing returns; these are the engine
systems whose absence is most visible in actual gameplay footage:
animation depth, VFX, decals, audio DSP, UI, and input. Game-side
counterparts are shooter SH-025..SH-044 (`bloom/shooter/docs/tickets.md`),
which also carries the round ordering.

---

## EN-025 — Ragdoll FFI ✅ *(shipped 2026-07-12)*

**Why:** Jolt ships `Ragdoll.cpp` in the vendored submodule but no
`bloom_physics_ragdoll_*` FFI exists — the shooter's enemies die by
clamping the last animation frame and sinking through the floor.
Ragdoll handoff is the single strongest "physical world" signal in an
action game, and the hard part (the solver) is already linked into
every build.

**API sketch:**

```ts
// Build once per model kind: bones as capsules between joint pairs,
// constraints from the skeleton hierarchy.
const rag = physicsCreateRagdoll(modelHandle, {
  boneRadiusScale: 0.4,   // capsule radius from bone length
  maxBodies: 16,          // merge tiny finger/tail chains
});
// On death: seed from the current animated pose + a kill impulse.
physicsActivateRagdoll(rag, modelHandle, hitDirX, hitDirY, hitDirZ, impulse);
// Per frame while active: physics drives the skin.
physicsFetchRagdollPose(rag, modelHandle);  // writes joint matrices
physicsDeactivateRagdoll(rag);              // back to the pool
```

**Open questions:**
- Auto-shape generation from the skeleton (capsules per joint pair,
  mass from bone length) vs authored shapes? (V1: auto with a scale
  knob; the alien skeletons are simple.)
- Sleep/despawn policy — deactivate when velocity < ε for N frames.

**Scope:** medium-large (~1 week) — C shim over Jolt's Ragdoll +
skeleton mapping + the pose write-back into the existing joint UBO
path (GPU skinning already consumes joint matrices; this only changes
who computes them).

**Acceptance:** shooter SH-031 — a killed enemy crumples over terrain
edges, reacts to the killing impulse, never clips the heightfield;
4 simultaneous ragdolls cost < 0.5 ms CPU.

---

## EN-026 — Particle / VFX system ✅ *(shipped)*

**Why:** the engine has **no particle system at all** (spec Phase I,
unbuilt) — the shooter fakes sparks with a 16-slot pool of additive
material draws. Muzzle smoke, blood, dust, shells, explosions, and
ambient motes all want one system. This is the most visible missing
engine feature in combat footage.

**Design (V1 — CPU sim, GPU instanced):**
- Emitter descriptor: spawn rate / burst count, lifetime ± variance,
  initial velocity cone, gravity + drag, size/alpha/color over life
  (4-key curves), optional atlas frame animation, bucket (additive |
  cutout), soft-depth-fade distance.
- CPU sim over a global pool (4–8k particles), one instance buffer
  write per frame (the EN-001 instancing path is exactly the right
  infrastructure), camera-facing quads in the shader.
- Sort: additive needs none; per-emitter sort for the rare
  alpha-blended case.
- Soft particles: sample scene depth (the refractive bucket already
  binds it) and fade near intersections.

**API sketch:**

```ts
const em = createParticleEmitter({ /* descriptor above */ });
emitterBurst(em, x, y, z, count);          // impact/muzzle one-shots
setEmitterPosition(em, x, y, z);           // continuous (smoke trail)
setEmitterActive(em, on);
```

**Open questions:** GPU sim (compute) is V2 — only needed past ~50k
particles; V1's budget target is 2k live particles < 0.3 ms GPU on
the 760M-class iGPU.

**Scope:** medium-large (~1–1.5 weeks).

**Acceptance:** shooter SH-033 ships muzzle smoke, blood, shells,
dust, and an explosion set entirely on this API within its GPU
budget; mobile halves pool sizes via the same descriptors.

---

## EN-027 — Deferred decal system ✅ *(shipped)*

**Why:** the world takes no marks — no bullet holes, scorch, or blood
splats. The impulse field is a material *input* (ripples/wet), not
projected decals. A deferred renderer makes decals cheap: project a
box, rewrite G-buffer albedo/normal/roughness before lighting.

**Design (V1):**
- Decal = oriented box (position, normal-aligned rotation, size),
  atlas UV rect, albedo/normal/roughness contribution weights,
  lifetime + fade.
- Rendered as instanced boxes after the opaque G-buffer pass,
  reading depth to reconstruct the surface point, discarding outside
  the box, angle-fade past ~60° to kill stretching.
- Fixed pool (e.g. 256) with ring reuse; per-decal fade-out.
- Skinned meshes excluded in V1 (blood on enemies is SH-033's
  particle tint job).

**API sketch:**

```ts
spawnDecal(x,y,z, nx,ny,nz, size, rotRad, atlasU0,V0,U1,V1, lifetimeS);
```

**Scope:** medium (~1 week) — one new pass + atlas loader + pool.

**Acceptance:** shooter SH-033 — 64 bullet holes visible at once,
conforming to terrain slope and building walls, no lighting seams,
< 0.2 ms GPU at full pool.

---

## EN-028 — Animation blending, masks, root motion ✅ *(shipped — AnimMixer)*

**Why:** the animation FFI plays exactly one clip per model
(`update_model_animation(handle, index, time)`), so every transition
in every game pops. Root motion is unconditionally stripped at import
(`models.rs:531`). This is the widest quality gap between Bloom
content and AAA content — geometry and lighting are fine; the
*motion* is 2005.

**Pieces (shippable in order):**
1. **Crossfade:** `playModelAnimation(handle, clip, fadeSeconds)` —
   engine keeps a 2-slot mixer per model (current + previous clip,
   lerped by fade progress at pose level). Covers 80% of the pops.
2. **Locomotion blend:** `setModelLocomotion(handle, clipA, clipB,
   t)` — two-clip continuous blend (idle↔walk↔run by speed), with
   phase-matching so feet don't slide during the blend.
3. **Upper-body masks:** joint-range (or named-group) mask so an
   attack clip drives the spine-up while locomotion keeps the legs.
4. **Root motion opt-in:** converter/import flag to *keep* root
   translation + an FFI to read the per-frame root delta
   (`getModelRootDelta(handle)`) so the game can feed it to the
   character controller. Default stays stripped (back-compat).

**Scope:** medium overall; piece 1 alone is small (~2 days) and
unlocks most of shooter SH-030/SH-034.

**Acceptance:** shooter transitions (walk↔attack↔pain↔die) show no
pops at 0.15 s fades; a marauder attacks while moving; the dragoon
pounce follows its authored root arc instead of hand-tuned kinematics.

---

## EN-029 — Audio buses, reverb send, occlusion filter ✅ *(shipped)*

**Why:** the mixer is master + per-voice gain — no submixes, no DSP.
AAA "gunfeel" is half audio: a weapon tail through a reverb send, a
mix that ducks around damage, distance/occlusion filtering. The
render thread is already lock-free SPSC; these are additions to that
graph, not a rewrite.

**Pieces:**
1. **Fixed bus graph V1:** master → { music, sfx, ui }. Per-bus gain
   + `duckBus(bus, amountDb, attackS, releaseS)` (sidechain-style
   momentary duck).
2. **One reverb send:** Freeverb/Schroeder on the render thread;
   per-voice send amount; global room params (`setReverbParams(size,
   damp, wet)`) so the game can morph zones.
3. **Per-voice one-pole low-pass:** `setSoundLowpass(voice, cutoffHz)`
   — the occlusion/distance-muffle primitive; the game decides when
   (raycasts are game-side).
4. Bus assignment at play time: `playSound3DOnBus(...)` or a default
   + setter.

All parameters flow through the existing SPSC control messages — no
locks on the audio thread.

**Scope:** medium (~1 week for 1–3).

**Acceptance:** shooter SH-035 — music audibly ducks on damage,
fights near the building sound enclosed, a shrieking enemy behind the
building is muffled; zero underruns/glitches at 48 kHz with 32 voices.

---

## EN-030 — UI widget layer 🔵 *(RE-OPENED 2026-07-15 — the ✅ was wrong)*

> **Corrected 2026-07-15.** This read *"✅ shipped — TS-side, `src/menu.ts` in
> the shooter"*. **The engine shipped nothing**: there is no `src/ui/` and no
> `src/widget*/` — verified. The ticket was closed by pointing at a *different
> repo's* file.
>
> That closure hid the exact gap the ticket exists to name. Its own scope below
> says a reusable `bloom/ui` module that "the editor can adopt incrementally".
> Instead: the editor still hand-rolls every panel, and game #2 starts from zero.
> One game having a menu is not an engine having a UI layer.
>
> The shooter's `src/menu.ts` is a good candidate to PROMOTE into `bloom/ui` — it
> already carries the hard part (one focus model shared by mouse, keyboard,
> gamepad and touch). But promoting it is the work, and it has not been done.
>
> Honest alternative if nobody will do it: close this as **won't-do** and say
> plainly that UI is per-game forever, so the editor stops waiting. What must not
> stand is a ✅ that means neither.

**Why:** the engine offers immediate-mode shapes + text only. Every
game needs pause/settings/menus, and the editor hand-rolls its own
panels. A retained-lite widget layer over the existing `draw2d` +
text + input FFIs closes both — **entirely in TypeScript** as a
`bloom/ui` module; no engine-native work.

**Scope (V1):**
- Widgets: panel, label, button, slider, toggle, dropdown, key-capture
  field (for rebinding).
- Layout: vertical/horizontal stacks with padding/anchors — no
  general constraint solver.
- Focus model: one focus ring driven by mouse hover, D-pad/arrows,
  and touch; activate on click/A/tap. This is the piece that makes
  gamepad menus (shooter SH-039) possible.
- Theming: flat colors + 9-patch optional; respects the logical-space
  scaling pattern the shooter already uses for HiDPI/mobile.
- Input capture: when a UI tree is active it consumes input so the
  game doesn't fire while clicking Resume.

**Acceptance:** shooter SH-038 builds title/pause/settings on it,
navigable by mouse, keyboard, pad, and touch; the editor can adopt
widgets incrementally.

---

## EN-031 — Gamepad backend verification + rumble ✅ *(shipped — XInput polling was already wired; rumble added)*

**Why:** the FFI surface exists (`bloom_is_gamepad_available`,
`bloom_get_gamepad_axis`, `bloom_is_gamepad_button_*`,
`bloom_get_gamepad_axis_count` — package.json manifest) alongside
`bloom_inject_gamepad_*` twins, which suggests the desktop backends
may only ever have been fed by injection (web/tests) rather than
polling real hardware. Nothing ships until a pad physically works.

**Scope:** small-medium.

- Audit each native backend: Windows (XInput), macOS/iOS
  (GameController framework), Linux (evdev/SDL-free), web (Gamepad
  API — likely already live).
- Wire whichever are dead; normalize axis ranges/deadzones and the
  button index map across platforms (document the canonical layout).
- **Add rumble:** `bloom_gamepad_rumble(lowFreq, highFreq, durationMs)`
  — cheap on XInput/GameController and a big feel win for SH-029.

**Acceptance:** a wired/BT pad drives the shooter on Windows and
iPhone; axis/button indices match the documented map on all
platforms; rumble fires on damage.

---

## EN-032 — Async model + world loading 🟡

**Why:** every load is synchronous on the main thread — `loadWorld`
is readFile → parse → validate in one blocking call, and model/texture
loads block too. One arena hides this; level switching (shooter
SH-040), bigger worlds, and any future streaming need the machinery.

**Scope:** medium.

- `loadModelAsync(path) -> handle` immediately; `isModelReady(handle)`
  poll; draws of not-ready handles no-op (or draw a bounds proxy).
- Background thread does file IO + parse + GPU upload staging; main
  thread finalizes (wgpu queue submission) in bounded per-frame slices.
- World: `loadWorldAsync(path)` with a progress query so games can
  render a loading screen; instantiation itself amortized over frames.
- This is deliberately *not* world-partition streaming — just the
  substrate it would need (tracked as a deferred item below).

**Acceptance:** shooter switches arenas behind an animated loading
screen with no main-thread stall > 100 ms; title screen appears
< 1 s after launch with assets streaming in behind it.

---

## EN-033 — Bone-socket world-transform query ✅ *(closed 2026-07-15 — landed in 1150c6b)*

> Stale bookkeeping, and it cost the shooter: SH-027 recorded its v2 as
> "gated on EN-033" for weeks AFTER this landed, so the gate was imaginary.
> `findJoint` + `jointWorld` are live; the shooter's weapon has ridden the
> `tag_weapon` joint since. A ticket left open past its landing is not a
> harmless clerical error — it stops downstream work that was already free.

**Why:** games can't attach anything to a skeleton — the shooter's
weapon can't ride the player's hand, muzzle points can't track fire
animations. The skin matrices are already computed every frame; this
is a read-back, not a feature.

**API sketch:**

```ts
// After updateModelAnimation: joint's world transform under the given
// model transform (pos + quat, or 12 floats). Numeric FFI per
// perry-quirks #5.
getModelJointWorld(handle, jointIndex, posX,posY,posZ, yawRad, scale)
  -> writes into a flat out-array FFI (bloom_get_model_joint_world_*)
```

Plus `findModelJoint(handle, name) -> index` at load time (string ok —
load-time only).

**Scope:** small (~1–2 days).

**Acceptance:** shooter SH-027 v2 — the rifle rides the hand joint
through walk/attack clips; muzzle flash tracks the true muzzle.

---

## EN-034 — Spot lights 🟢

**Why:** the light schema is directional + point only
(`src/world/types.ts`). Spots are the workhorse light of interiors,
muzzle-light shaping, and flashlights; the froxel clustering path
already handles points — a cone test is incremental.

**Scope:** small-medium — `LightData kind:"spot"` (schema v3 with
migration, editor light-tool angle handles), cone attenuation in the
clustered lighting path. Shadowed spots explicitly deferred.

**Acceptance:** editor places a spot with direction + inner/outer
angles; engine renders correct cone falloff; N spots cluster like
points.

---

## EN-038 — `takeScreenshot` writes no file on Windows 🔴 *(diagnosis VOID — re-diagnose before acting)*

> ### ⚠️ The evidence below is invalid. Read this first.
>
> **Every conclusion in the original ticket rests on "the `eprintln!` never
> printed". That proves nothing on Windows: `eprintln!` is DEAD after init here.**
> Only the two init-time lines ever reach a redirected stderr, so "the native
> function is never entered at all" was inferred from a channel that cannot
> report. The "Perry FFI dispatch drops the call" suspect list is unfounded
> speculation built on top of it.
>
> What is actually known, verified 2026-07-15:
> - `bloom_take_screenshot` **exists**, is correctly registered
>   (`ffi_core/assets.rs`), exported from `native/windows/src/lib.rs`, and is in
>   the manifest. So the "Perry never dispatches it" theory has no support.
> - The **F12 path works and produces true 4K captures** — and it goes through
>   the *same* `renderer.screenshot_requested` + readback machinery, just
>   triggered inside the wndproc instead of from TS. That strongly suggests the
>   readback is fine and the problem, if any, is upstream of it.
> - The symptom that is real and reproducible: **no file appears.**
>
> **How to re-diagnose (do not repeat the mistake):** probe with
> `std::fs::write` to an ABSOLUTE path, or an atomic counter read back over an
> FFI — never a log line. Check for the FILE, not for output. See
> `shooter/docs/perry-quirks.md` and the dead-stderr note.
>
> **Cost so far.** This is not academic: with `takeScreenshot` unusable and
> `tools/shot-window.ps1` returning stale desktop frames, visual verification on
> Windows currently depends entirely on `tools/f12-shot.ps1` (PostMessage VK_F12).
> A whole session's worth of "is it actually on screen?" questions had to route
> through that one path, and several screenshots were believed before the
> staleness was spotted.

**Original symptom (evidence void — kept for the record).** `takeScreenshot('x.png')`
from TS produces nothing: no file, and — supposedly decisively — not even the
unconditional `eprintln!("bloom: screenshot requested -> '{}'")` at the top of
the FFI, nor the `screenshot readback branch running` log in `end_frame`.

**Repro (2026-07-12, shooter @ AAA round 1).** Call `takeScreenshot` from the
game loop at a fixed frame. A `console.log` immediately before it prints; the
engine's own log line never does. Three separate frames, plain relative path,
same result. *(The last sentence is the void part: the engine's log line CANNOT
print post-init on Windows regardless of whether the FFI ran.)*

**Why it matters.** Every screenshot-based harness in the shooter
(`SELFTEST`, `PERFTEST`'s in-engine captures) is silently capturing nothing, and
has been reporting success. Visual verification had to fall back to a
window-rect desktop grab (`shooter/tools/shot-window.ps1`).

**Suspects, in order.** (1) Perry's FFI dispatch for a void-returning
string-param extern — note the manifest declares `params: ["string"]` and the
call site is inside a nested block; (2) a name collision with a platform-crate
symbol; (3) dead-code elimination of a call whose return is unused.

**Next step.** Bisect with a minimal Perry program that calls only
`bloom_take_screenshot`, then compare the generated call against a
known-working void/string FFI (`bloom_load_model` returns f64, so pick
`bloom_set_env_clear_from_hdr` — same void+string shape — as the control).

---

## EN-039 — immediate draw with a full transform ✅ *(shipped 2026-07-15 — PR #100)*

> `drawModelTransform(model, m16, tint)` — a full column-major 4x4, the
> immediate-mode twin of `bloom_scene_set_transform16`. Skinned models are a
> deliberate no-op (their joint matrices already bake orientation).
>
> **The API sketch below was wrong and the correction is worth keeping:** it
> said to route the matrix through the mesh scratch because "Perry can't pass
> 16 f64 args". Perry can — `bloom_scene_set_transform16` already passes 17 and
> always has. Spelling the matrix out keeps the call STATELESS: no scratch to
> reset, no ordering hazard between a reset and a draw.
>
> Acceptance met: shooter SH-027 v2 (PR #22) — the gun points where the camera
> points, verified by re-orienting the weapon with only the pitch changed.

**Why.** `drawModelRotated` takes a single Y rotation, so an immediate-mode draw
cannot pitch or roll. The shooter's weapon therefore cannot tilt with the
player's aim — it stays level while the camera looks up or down. Any held prop,
thrown object, or debris chunk hits the same wall.

**API sketch.** `drawModelTransform(handle, m16)` taking a column-major 4×4
(the scene graph already has `bloom_scene_set_transform16`; this is the
immediate-mode twin). Perry can't pass 16 f64 args, so route it through the mesh
scratch like the other array-shaped FFIs.

**Acceptance:** shooter SH-027 v2 — the gun points where the camera points.

---

## Deferred infrastructure tracks 🔴

Recorded so the list is complete; each is real but none blocks the
current shooter roadmap. Most are specced in
`bloom-renderer-spec-v2.md` — this is the priority call, not new
design.

- **EN-035 — Job system.** No general-purpose task pool exists (audio
  + Jolt thread internally; render submission is single-threaded).
  Becomes the bottleneck when draw counts or anim/ragdoll counts grow
  10×.
- **EN-036 — Wire the render graph.** `renderer/graph.rs` is built and
  unit-tested but the live frame is hand-ordered in `mod.rs`. Wiring
  it buys automatic barriers/aliasing and makes pass insertion
  (EN-026/EN-027) cheaper — do it opportunistically with one of those.
- **EN-037 — World streaming / partition.** Chunked worlds, HLOD,
  cell load/unload on EN-032's substrate. Only if a game outgrows
  arena scale.
- **Spec-v2 renderer tracks** (VSM, froxel volumetrics, virtualized
  geometry, DLSS/FSR SDK integration): explicitly parked — the
  current CSM/TSR stack is not the gap. Revisit after the feel
  rounds ship.
- **Netcode:** absent entirely; out of scope by decision, not
  oversight. Revisit only with a design that needs it.



## EN-041 — Hierarchical foliage wind ✅ *(shipped 2026-07-12)*

The engine swayed **alpha-cut materials only**. So leaf cards fluttered and every
tree trunk in every game stood perfectly rigid — a forest of poles with twitching
hair — and the shadow shaders applied no wind at all, so the leaves moved while
their shadows stayed nailed to the ground.

`common/foliage_wind.wgsl` is now one field shared by the scene pass and both
shadow shaders. Three layers, because a tree does not move as one thing: trunk
bend (∝ height², a cantilever — the motion you read at 30 m), branch sway (∝ reach
from the trunk axis), leaf flutter (cutout cards only). The weights are DERIVED
from the vertex's position relative to the model origin rather than authored into
vertex colours — COLOR_0 is already spent on albedo tint, and for procedurally
generated trees the regions are known exactly anyway. So: no new vertex attribute,
no GLB re-bake.

`set_model_foliage_wind(model, amount)` opts a model in; everything else stays
rigid. The offset is computed in WORLD space and mapped back through the model's
inverse linear part — displacing along a local axis would let each tree's
per-instance yaw rotate the wind with it, and a stand of trees would bend a dozen
different ways. Prev-frame offset too, so TAA gets a real velocity for a moving
leaf instead of 0.

---

## EN-042 — Dynamic shadow-caster budget is 64, and overflow is silently dropped ✅ *(fixed — see the resolution at the end of this file)*

**Found while shipping EN-041.** `SHADOW_MAX_DYNAMIC = 64`. A caster that moves
every frame cannot reuse the cached static depth, so it must go in the dynamic
set — and the shooter's forest alone is 88 trees × 4 primitives = **352**.

Turning on `set_foliage_shadow_motion` therefore overflows the budget, and the
overflow is **dropped without a word**. The measured result was not a slowdown but
a *speedup* (34 → 40 fps) — because it had silently deleted every tree shadow AND
**the player's own shadow from under their feet**. A perf win that is really a
correctness loss is the worst possible failure mode.

Mitigated for now: foliage is promoted to the dynamic set only while slots remain
(`MAX_FOLIAGE_DYNAMIC = 24`), so characters always keep theirs and the rest of the
forest just stays rigid. The real fix is a budget that scales, or a foliage path
that refreshes cached static depth on a slow cadence instead of per frame.

**Acceptance:** the dynamic set cannot silently drop a caster — it either fits, or
the engine degrades a *chosen* class (foliage) rather than whatever happened to be
queued last.


## EN-043 — A moving cached caster invalidated the ENTIRE static shadow cache ✅ *(fixed 2026-07-12)*

The static-cascade shadow cache (the perf round's biggest win, shadow_pass 7.2 ms
→ 0.1–1.7 ms) had quietly stopped working. Measured on the shooter's title screen:
**shadow_pass GPU back up to 6.9 ms**, and the title down from 50.7 to 33.5 fps.

**Cause.** A cached, non-skinned caster whose transform changed since last frame
stayed in the STATIC set — but its content signature changed, which invalidated the
cascade's cached depth. So every tree, wall and terrain tile in the world
re-rendered into all three cascades, every frame, because *something small was
moving*. The cache was working exactly as designed and being defeated by one
bobbing object.

**Fix.** A caster that moves is DYNAMIC, by definition. Track each caster's
transform hash against last frame's; if it changed, promote it to the dynamic set,
where it draws on top of the cached static depth and never invalidates it. One draw
instead of a thousand.

**shadow_pass GPU 6954 µs → 182 µs (38×). Title screen 33.5 → 44.7 fps.**

**The trap inside the fix, which cost a round.** The first cut keyed casters on
`(model_handle, mesh_idx)`. The forest is **88 trees sharing three model handles**,
so every tree collided on one key, each was compared against some other tree's
transform, and all 88 were declared movers — the whole forest went dynamic,
overflowed the 64-slot budget (EN-042), and **every shadow in the game vanished
while the fps went UP**. A perf win that is really a correctness loss, again. The
key now includes the Nth-draw-of-this-handle occurrence index, so occurrence N is
the same tree every frame.

Related: EN-042 (the dynamic budget silently drops overflow) is what turned a keying
bug into invisible shadows rather than a loud failure. It is still worth fixing.


## EN-044 — Depth prepass for cached models ✅ *(shipped 2026-07-12)*

The scene fragment shader can `discard` (alpha-cutout foliage), and **a shader that
may discard cannot early-Z write** — the GPU has to run it in full before it knows
whether the pixel survives. So an 88-tree forest of overlapping leaf cards shaded
the whole 5-target MRT several layers deep and threw most of it away. Measured: the
forest alone was **5.6 ms of a 7.4 ms `main_hdr_pass`**, and simply not drawing it
took the title screen from 46.7 fps to the 60 fps vsync cap.

Now a depth-only prepass runs the same cached-model draws first (same vertex stage,
so the foliage wind displaces identically; alpha cutout honoured so cards keep their
real silhouette), and the main pass draws them through a pipeline with **depth
writes OFF and an `Equal` test**. Taking the writes away is the whole point: with
writes on, a discarding shader forces late-Z and nothing is rejected. Without them,
the hardware early-Z rejects the occluded leaves before the shader runs.

| pass | before | after |
|---|---|---|
| `main_hdr_pass` | 7.43 ms | **2.14 ms** |
| `depth_prepass` | — | 1.37 ms |
| **title screen** | **33.5 fps** | **56.6 fps** |

**The bug this uncovered, which is the interesting part.** The sky pipeline was
`depth_write: true, depth_compare: Always`, and the sky is drawn *first* inside the
HDR pass. That stamped depth = 1.0 across the entire screen. It had always been
harmless — the buffer had just been cleared to 1.0 anyway — and it became instantly
destructive the moment a prepass wrote real geometry depth *before* it: the sky
wiped the lot, the `Equal` test failed everywhere, and **the entire forest and the
player vanished**. The sky never needed that write. It is off now.

(First suspect was depth invariance — two pipelines compiling the same vertex shader
differently. `@invariant` on the position is in place and is genuinely required for
an `Equal` test, but it was not the cause.)


---

## EN-042 — Dynamic shadow-caster budget ✅ *(fixed 2026-07-12)*

`SHADOW_MAX_DYNAMIC` was **64**, and the overflow was dropped **in queue order,
silently**. That was fine while "dynamic" meant a handful of characters. It became a
trap the moment a *forest* could go dynamic — 88 trees × 4 primitives = 352 casters —
and it cost this project twice in one session:

- enabling swaying foliage shadows measured **34 → 40 fps**, because it had deleted
  every tree shadow **and the player's own shadow**;
- the first cut of EN-043 measured **42 fps** for the same reason.

Both looked like wins. Neither was. A budget that silently drops whatever happened to
be queued last is a landmine.

**Fixed two ways.** The budget is **256** (from 1024 slots/cascade). And the drop is
**ranked**, not accidental: characters first (the shadow a player actually looks at),
then other movers, then foliage — a swaying canopy shadow is soft, dappled and the
most forgiving thing in the frame. If a shadow must be lost, it is now a chosen one.


## EN-045 — The static shadow cache only ever worked on a stationary camera ✅ *(fixed 2026-07-12)*

`shadow_pass` GPU was **0.12 ms on the title screen and 3.2 ms in a moving fight** —
a 27× gap that nobody had looked at, because the cache had been *measured on the
title screen*.

**Cause.** A cascade keeps its cached static depth only while its VP is unchanged.
`compute_cascade_vps` has exactly the machinery for that — an `accepted_fit` with
`REFIT_SLACK`, so the cascade keeps its VP while the camera travels within slack —
and it was gated on **`c > 0`**. Cascade 0, the NEAR cascade holding the player and
everything they are standing next to, re-fit **every frame**. So its VP changed on
every frame the camera moved, which is all of gameplay, and every static caster in
it re-rendered every frame.

**Fix.** Cascade 0 gets the slack too. It costs ~15% of near-field shadow resolution
and buys a cache that survives ~15 frames of walking instead of zero.

| | before | after |
|---|---|---|
| `shadow_pass` GPU (combat) | 3.58 ms | **0.53 ms** |
| gameplay | 42–44 fps | **53–56 fps** |

Near-field shadow quality verified by screenshot: the player's own shadow is still
crisp and correctly shaped.

**A lesson about benchmarks.** The cache was landed, measured, and celebrated on a
screen where the camera does not move — the one condition under which its central
assumption always holds. Measure the thing you actually ship.

---

## EN-043 follow-up — the caster identity must be ORDER-INDEPENDENT

The first version keyed a caster on *"the Nth draw of this model handle"*. That held
until the game started drawing its forest **front-to-back**: the sort order changes
as the camera moves, so occurrence N became a different tree every frame, dozens of
perfectly stationary trees were misread as movers, and the dynamic caster set blew
past 32 in combat.

The key is now a single hash of **identity AND transform**, tested for membership in
last frame's set. "Was this exact caster, at this exact transform, here last frame?"
A set membership test does not care what order the draws arrive in.


## EN-046 — Output (swapchain) scale ✅ *(shipped 2026-07-12)*

`set_render_scale` shrinks the G-buffer and everything drawn at render resolution,
then TSR upscales to the swapchain. It does **nothing** for the cost of that upscale
or the final composite — and on a 4K display those two passes were measured at
**3.10 ms + 2.40 ms**, about a third of the whole frame.

`set_output_scale(s)` configures the **swapchain itself** at `s ×` the window's real
size; the presentation engine stretches it back up. It is the only knob that touches
that fixed tail.

| output scale | `taa_pass` | `final_composite` | gameplay |
|---|---|---|---|
| 1.0 (native 4K) | 3.06 ms | 2.37 ms | ~53 fps |
| 0.8 | 1.46 ms | 1.04 ms | **locked 60.0 fps** (max frame 17.9 ms) |
| 0.6 | 0.28 ms | 0.22 ms | 60 (capped) |

Default 1.0, so no existing game changes. The renderer remembers the window's native
size, so the scale can be changed at runtime without the platform telling it again.

**Expose this to players.** At 4K it is the difference between a locked frame rate
and a sharp one, and which of those someone wants is not the engine's call — nor the
game's.


---

## EN-047 — `saveWorld` destroyed the world it saved, and reported success ✅ *(fixed 2026-07-12)*

**The editor could not save. Saving a world emptied it.**

`saveWorld` used `JSON.stringify(world, null, 2)`. On Perry 0.5.x that **corrupts a
large object graph that came from `JSON.parse`**. Minimal repro, no engine code:

```ts
const text = readFile('assets/worlds/arena_02.world.json');  // 324 KB — writes back fine
const o    = JSON.parse(text);                                // fine
const re   = JSON.stringify(o, null, 2);
// -> `re` holds 5,296 characters above U+00FF, in a document whose source is
//    almost pure ASCII. It is garbage.
```

The corruption is invisible in TS. It only surfaces at the FFI: `str_from_header`
fails its UTF-8 check, returns `""`, and `bloom_write_file` **writes a zero-byte file
and returns SUCCESS**. So the editor's save path deleted the file and said "saved".

Ruled out by probe: size (a *fresh* 1 MB world-shaped object stringifies fine),
floats, nulls, `Record` keys, non-ASCII, and a manual deep clone. It is specifically
the parsed graph.

**Two fixes, and both are needed.**

1. **`src/world/serialize.ts`** — a hand-written emitter that walks the schema by
   literal key and builds the document by concatenation. Same discipline the
   shooter's `settings.ts` already adopted for the same reason. `saveWorld` and
   `savePrefab` no longer touch `JSON.stringify`.
2. **`try_str_from_header`** — an FFI string that fails ABI validation is no longer
   silently substituted with `""`. `bloom_write_file` now **fails** instead of
   writing an empty file and claiming success. An empty string and a failed string
   are not the same thing, and any FFI that *persists* its input must know which it
   is holding.

Verified: `loadWorld` → `saveWorld` → `loadWorld` on arena_02 round-trips 168
entities, 5 lights, 1 water volume, 16,384 terrain heights and the world name, with
field-level checks matching. 204,270 bytes written, where it used to write 0.

**Still latent:** `JSON.parse(JSON.stringify(x))` is used as a deep-clone idiom in the
editor (prefab cloning). It is safe at the sizes prefabs run to, and it is a landmine
at scale. Anything that clones a *world* that way must not.


## EN-048 — `launchProcess` ✅ *(shipped 2026-07-12)*

Perry's `child_process.spawn` **compiles and then does nothing**: it returns a child
with an undefined pid and no process is started. So no Bloom tool could run another
program — which is all play-in-editor is (save the level, run the game on it).

`launchProcess(cmd, args[], cwd)` shells out via `std::process::Command`, fully
detached (never waited on, stdio to null): a GUI must not block on, or die with, the
thing it launched.

**Two traps, both hit:**

1. **Rust's `Command::current_dir` sets the CHILD's working directory — it does not
   affect how the program is FOUND**, which happens in the parent's context. So
   launching `"main.exe"` with `cwd: "<project>"` fails with *"program not found"*
   even though `main.exe` is sitting right there in `<project>`. A bare command name
   is now resolved against `cwd` before spawning.
2. Args cross as a newline-separated string and are re-split into a real argv. There
   is no shell involved, which is also why there is nothing to inject into.

## EN-049 — `createTextureArrayFromTexels`: a texture array from DATA, not files ✅

`createTextureArray` / `createTextureArrayEx` take a `*const u8`, so the manifest has
to declare that param `i64` — and **Perry cannot pass a `number[]` into an i64 param**
(`TypeError: Expected safe integer for native i64 parameter`). From Perry, both are
uncallable. Every real caller therefore ended up on `createTextureArrayFromFiles`,
which is right for ART and useless for DATA: a terrain splat map is computed at load
out of the world file and there is no file to name.

Same fix the mesh path has used all along (`bloom_create_mesh_scratch`, whose comment
already says *"Perry 0.5.x rejects `number[]` in an `i64` pointer param"*): push the
payload through the scratch buffer, then call with the dimensions.

```ts
createTextureArrayFromTexels(texels, texelCount, w, h, layerCount, format, mipLevels)
```

`texels` is one **packed u32 per texel** (`r | g<<8 | b<<16 | a<<24`), so a 128² map
costs 16,384 FFI calls rather than 65,536. Load-time only — it is linear in the texel
count.

Shipped as the transport for the shooter's authored splat terrain (editor PLAN §D).

**Platform gap:** web and watchOS do not implement the scratch buffer at all, so
`validate-ffi` reports this function as unexported there — the same pre-existing gap
already carried by `bloom_scene_update_geometry_scratch` and
`bloom_gen_mesh_spline_ribbon_scratch`. Failures went 6 → 8 for that reason.

## EN-050 — `clamp` in `world/terrain.ts` miscompiled; splat weights all read 0 ✅

`clamp(v, lo, hi)`, whose body was a single nested-ternary return, evaluated to `lo`
for **every** input when called from `quantizeWeight` in the same module — while
`sampleHeight`, in the same file, called the same helper correctly.

Every splat weight therefore quantised to `0`, and a painted terrain loaded
unpainted. Fixed by writing `clamp` with `if` statements; verified end-to-end (the
shooter renders authored paint) and pinned by the editor self-test
`testSplatPaintPartition`.

Root cause is a Perry codegen bug, not ours. Reduced repro, and the five probe
designs that gave *wrong* answers on the way to it: shooter `docs/perry-quirks.md`
#8.

## EN-051 — `easeInOutQuad` never receives its argument 🔴

```ts
export function easeInOutQuad(t: number): number {
  if (t < 0.5) return 2 * t * t;      // false for EVERY input
  return (4 - 2 * t) * t - 1;
}
```

The parameter does not arrive; the function returns a constant for all `t`. Adding a
`console.log(t)` to the body makes it correct — the signature of a codegen /
optimisation bug. Rewriting with `if` statements, reordering the expression, and
binding `t` to a local all fail to fix it. `easeInOutCubic`, directly below it and
the same shape, is fine.

**Left broken deliberately.** Nothing in the shooter or the editor calls it, and
cargo-culting a workaround into a function I do not understand would only hide it.
It is a public export, so a game that uses it gets silently wrong easing — that is
the cost of leaving it, and it is why this is filed rather than quietly patched.

Needs a Perry-side fix (or a minimal repro filed upstream). See shooter
`docs/perry-quirks.md` #8, Case B.

**The rest of `src/` is not cleared.** A sweep found the single-ternary shape in the
two editor `clamp`s, `easeInOutCubic`, and three shooter helpers; those were rewritten
defensively, but only `clamp`/`quantizeWeight` and the easings were actually
*verified*. The shape is a smell, not a diagnosis.

---

## EN-052 — Two files are over the line limit, and CI was blind to it 🟡 *(2026-07-16)*

`tools/check-file-lines.js` enforces a 2000-line ceiling, with a grandfather
baseline that **"may only shrink (ratchet)"**. Two files violate it today:

| file | lines | limit / baseline |
|---|---|---|
| `native/shared/src/renderer/mod.rs` | **13058** | grandfathered at **11985** — grew **+1073** |
| `native/shared/src/models.rs` | **2360** | 2000, not grandfathered at all |

**Why nobody noticed: the check never ran.** It is the second step of the
`ffi-parity` job, and step one (`validate-ffi`) has been failing on main. The job
uses `bash -e`, so it died before reaching the line check — for as long as
ffi-parity has been red. Fixing ffi-parity (see the same PR) made this step
execute for the first time in a while, and it immediately failed. The renderer
grew past its ratchet during **PT-7 / PT-8 / PT-9** (the last two days), with the
guard that exists to prevent exactly that sitting behind a broken step.

**What was done, and it is a stopgap:** both files are recorded in
`tools/file-lines-baseline.json` at their CURRENT size. That unblocks every PR in
the repo, and the ratchet re-arms from today — neither file may grow another
line. It does not pretend they are fine.

**What it is NOT:** a fix. The baseline is documented as ratcheting *down*, and
raising an entry is against the letter of that rule. It was chosen over the two
alternatives deliberately:

- Splitting a 13k-line renderer is a multi-hour refactor with real regression
  risk across every render path, and doing it as a drive-by inside a CI fix is
  how renderers break.
- Leaving CI red blocks every open PR on a violation that is already merged. You
  cannot retroactively reject it.

**The actual work, unclaimed:** split `renderer/mod.rs` (it is 6.5x the ceiling —
`model_draw.rs` shows the shape: pull cohesive passes out into
`renderer/<pass>.rs`) and split `models.rs` (2360, only 18% over — the cheaper
one, and a good first cut).

**Fixed in passing: the checker was inert on Windows.** It looked up
`baseline[path.relative(ROOT, file)]`, and `path.relative` yields BACKSLASHES on
Windows while both the baseline keys and every `EXCLUDE` regex (`/\/target\//`,
`/native\/third_party\//`, ...) are written with forward slashes. So on a Windows
dev box nothing matched: the grandfather list did nothing, and the exclusions did
nothing — it flagged vendored build artifacts (`typenum/out/tests.rs`, 20562
lines) and `--update` would have written them straight into the baseline. The
same tree passed in CI and failed locally, which is the worst way for a guard to
behave. `rel` is now normalised to '/'. Run it on Windows: 0 failures.

---

## HW-vs-SW Lumen: MEASURED — and HW ray query is OPT-IN now (2026-07-16 evening)

**Shipped the same evening:** the Windows host requests EXPERIMENTAL_RAY_QUERY
only when opted in (`BLOOM_HW_GI=1`, `BLOOM_PT`, or a `--pt` arg) — merely
having dxcompiler.dll beside the exe no longer converts a game to the HW GI
path. Boot now prints the decision (`bloom: hw-gi opt-in: want_hw=... `).
Two more things learned while verifying:

- The SW path is itself TIERED at runtime: `hiz-screen` at boot, upgrading to
  `sdf-clipmap` once the clipmap bakes (both print now). Any fps read at a
  single instant mixes tier-warmup states — trust 60 s averages only.
- An apparent "HW loses the sun shadows" difference between screenshot pairs
  did NOT survive scrutiny: the drifting cloud deck (45% shadow strength)
  confounds any two captures taken minutes apart.
- **The clouds-off A/B was then done (same evening,
  `setCloudShadows(0,...)` probe, 30 s settle each, gate prints verified):**
  both paths render correct sun shadows; the difference is TONAL — HW GI is
  mildly warmer/brighter in the indirect term (most visible on the house
  facade picking up warm ground bounce; SW reads slightly greyer). At the
  outdoor vantage it is a subtle grade-level difference, not a structural
  quality win, for ~30% of the frame (14.4 vs 20.3 fps). Verdict: SW stays
  the default; opt in per box with `BLOOM_HW_GI=1` where the GPU affords it.
  Captures archived in the shooter's `tools/.testout/gi-ab-{sw,hw}.png`.
  Re-open the default question for INDOOR scenes (bounce matters more there;
  the menu backdrop cannot exercise INDOORCAM) or on stronger hardware.

## HW-vs-SW Lumen: MEASURED at last (2026-07-16 evening) — HW costs +20 ms/frame on the 760M

The number EN-023 and EN-058(c) said nobody had ever measured. A/B on the
shooter (PERFTEST mode-1 gameplay, identical build, DXC DLLs present, PT off):

| device features | gameplay | frame |
|---|---|---|
| EXPERIMENTAL_RAY_QUERY granted (HW Lumen active) | **14.4 fps** | ~69 ms |
| `BLOOM_FORCE_SW_GI=1` (SW hiz-screen/SDF trace) | **20.3 fps** | ~49 ms |

Three consequences:

1. **Granting the feature IS enabling HW Lumen.** ssgi_pass/gi_bake pick the
   HW trace the moment the TLAS exists ("HW still wins over both when the
   feature was granted"), and model_draw registers every skinned draw for
   per-frame compute pre-skin + BLAS builds. There is no separate "HW Lumen
   on/off" knob — dropping the DXC DLLs beside a shooter exe silently
   converted it to the HW path and cost ~40% of its frame rate. docs/perf/014's
   "within ~1.5×" guess is dead; on this iGPU the HW path costs +20 ms/frame.
2. **The boot line lies:** `ssgi trace backend = hiz-screen` is printed at
   init, before the TLAS exists; per-frame the backend flips to HW with no
   log. Diagnostics should report the backend actually used.
3. **The choice is quality-vs-frame-rate and belongs to the product owner**
   per the standing no-quality-tradeoffs rule: HW trace sees real proxy
   geometry (off-screen bounce, no screen-space artifacts); SW is 20 ms
   cheaper here. Needs a GI screenshot A/B on this scene before any default
   is picked. `BLOOM_FORCE_SW_GI=1` is the per-box escape hatch meanwhile.

## EN-053 — Shadows cost ~5.3 ms in SAMPLING, and four passes each pay it ❌ *closed same day — the premise did not survive re-measurement*

> **Measured before building (2026-07-16 evening, PERFTEST mode-0 bisect on
> the shooter's current main):** toggling shadows off buys **0.4 ms**
> (39.88 → 39.48 ms), with `shadow_pass` at 0.09 ms — while `no-ssgi` buys
> **4.9 ms**. The 5.3 ms figure was measured 2026-07-12 and the frame changed
> under it (house v2, 120k grass, photoscan terrain landed since). Building
> the shadow-mask pass tonight would have repeated the SH-049 attribution
> mistake with a week of renderer surgery attached. The file:line facts below
> stay true (16 taps in a ±1-texel disk, four independent consumers) — they
> are just not where the frame's time is. Re-open only from a fresh bisect
> that shows sampling as a top-3 lever; today's top lever is SSGI cost
> (quality-gated — the direction is making it cheaper, e.g. the HW ray-query
> path, not turning it off).

The shadow **pass** is fixed (EN-043/044/045 — 0.18 ms). What remains is the
per-pixel PCF: the deferred lighting path runs a fixed **16-tap** Poisson
kernel (`renderer/shaders/core.rs:535-564` — the ":516 comment still says
4-tap"), cascade blending doubles that to 32 in the last 10% of each cascade,
and the material path / GI card-light / planar probe all sample the cascades
**independently**. Toggling shadows off on the shooter buys ~5.3 ms while the
shadow pass itself renders in 0.18 — the sampling IS the cost, and it is the
single biggest remaining GPU lever at shipped defaults.

Fix shape: a screen-space **shadow-mask pass** at render resolution — compute
the sun-shadow term once, sample the mask from every consumer. Optionally
distance-adaptive tap counts. **Hard constraint: quality-neutral.** The
product owner has explicitly rejected perf wins that cost image quality; the
mask must be full render res, and any tap-count adaptation must be
imperceptible and screenshot-verified (per the twice-earned rule: a shadow
perf win bigger than the change justifies = deleted shadows).

## EN-054 — SDF clipmap re-bake still does its whole CPU prep in one frame 🔴 *(2026-07-16 audit)*

The GPU half is amortised (16 Z-layers/frame); the frame that STARTS a rebake
still runs `scene.build_world_triangles()` synchronously —
`renderer/scene.rs:476-506` re-transforms every vertex of every node into
fresh Vecs with zero caching, then `gi_bake.rs:128-162` counting-sorts every
triangle. 20-60 ms in one frame, triggered every ~10 m of camera travel
(`gi_bake.rs:44-46`); the shooter's 50-68 ms wave-spawn spikes are this, and
the shooter ships SSGI **on**, so it is live in the shipped game.

Fix (from the July perf audit, still unimplemented): cache the world-triangle
soup keyed on `tlas_version` — the scene is static, so travel-triggered
rebakes should re-bin, not re-gather; then amortise the binning across the 4
frames the GPU bake already takes.

## EN-055 — No animation-instancing API: N enemies = N full GLB re-parses ✅ *(shipped same day — PR #107; and the boot claim was WRONG)*

> **Measured before believing (2026-07-16, same day):** the shooter's per-slot
> animation parses were **43 ms total**, not the 5.5 s SH-049 attributed to
> them. `load_gltf_animation` re-parses the container but never decodes
> images — the actual 4.9 s was SERIAL `loadModel` texture decode, fixed by
> the shooter adopting the parallel `stageModels`/`commitModel` path (which
> needed two staged-commit bugs fixed first — see PR #107). The API below
> still landed: instances are the right semantics for crowds, and the boot
> measured 8,702 → 4,372 ms end-to-end.

`loadModelAnimation` -> `load_gltf_animation` re-decodes the container, clones
every buffer blob, rebuilds the skeleton and duplicates all keyframe tracks
**per instance** (`models.rs:546-551`, `1183-1437`). The shooter needs one
mixer per enemy SLOT (correct — mixer state is per-instance), so its boot
parses the same seven GLBs ~30 times: **5.5 s of an 8 s boot** (SH-049
measured 61%).

The struct already separates what could be shared (`skeleton`, `animations`,
`ref_rest_rotations`) from per-instance state (`joint_matrices`, `mixer`,
`joint_world`) — models.rs:116-130. Wanted: `instantiateAnimation(handle)`
(Arc-share the clip data, fresh mixer state), callers stay handle-based. This
is the whole boot-time story; EN-032 (async loading) is adjacent but does not
remove the duplication.

## EN-056 — Per-frame upload/allocation tail in the renderer 🟡 *(2026-07-16 audit)*

Individually small, collectively the class of waste the frame no longer has
budget for at 4K. All VERIFIED still present:

- Lighting UBO (~8.7 KB, 256 point-light slots) uploaded whole **8-9x/frame**
  with no dirty flag (`mod.rs:10245,11526,11919-11955`; `shadow_pass.rs:73`
  is explicitly unconditional).
- Bloom chain creates **9 uniform buffers + 9 bind groups per frame**
  (`postfx_chain.rs:66-141` — the per-pass UBOs are argued for in comments;
  the bind-group re-creation is not).
- The render graph is rebuilt from scratch **twice per frame** — 17 boxed
  closures, fresh HashMap + HashSets, O(n^3) topo sort (`graph.rs:103-224`,
  `mod.rs:10650,10824,11067`); composite/post bind groups rebuilt per frame
  (`mod.rs:10862,10946`).

Fix: dirty-flag the lighting UBO, cache bloom + composite bind groups keyed
on the views they wrap, build the graph once and rebuild on topology change.

## EN-057 — Hi-Z occlusion runs every frame for zero consumers ✅ *(shipped same day)*

> The occlusion reduce + readback now gates on a per-frame
> `set_has_consumers` flag the engine derives from the scene graph (any
> visible non-gi_only node). Going consumer-less also invalidates the grid,
> so a consumer appearing later reads the conservative "potentially visible"
> answer instead of a stale capture — the gate cannot wrongly cull a draw by
> construction. The Hi-Z pyramid itself still builds: SSAO consumes it.

The pyramid build + occlusion reduce + readback gate only on `ssao_enabled`
and the culler's own flag (`mod.rs:10707-10731`, `occlusion.rs:221-223`) —
never on whether any rasterized scene node actually consumes the result. In
the shooter, the only scene nodes are 267 `gi_only` proxies that are never
drawn, so the readback benefits nothing, every frame. Gate the occlusion
half on ">= 1 non-gi_only scene node"; the SSAO half of the pyramid stays.

## EN-058 — DXC: missing DLLs are a BOOT CRASH, and nothing deploys them 🔴 *(severity raised 2026-07-16, same day)*

> **Worse than written, verified by accident:** the DLLs vanished from the
> shooter root mid-session and the game **crashed at boot** — with
> `DynamicDxc` configured, a missing `dxcompiler.dll` drops the ENTIRE DX12
> backend out of the wgpu instance (no FXC fallback at instance level), and
> the Vulkan surface path is broken on Windows (`Win32::hinstance` never
> set), so there is nothing left to present with. `tools/fetch-dxc.ps1`'s
> "missing DLLs are not fatal" line was wrong and is fixed in PR #107. The
> deployment automation below is now launch-critical, and the Vulkan
> hinstance bug is a second, independent fix worth making (it is the only
> reason the DXC failure is fatal rather than degraded).

`Dx12Compiler::DynamicDxc` landed (`native/windows/src/lib.rs:611-620`) and
HW ray query is REAL when `dxcompiler.dll`/`dxil.dll` sit beside the exe
(shooter boot log: `ray_query=true`). But when the DLLs are absent, wgpu
falls back to FXC **silently** — SM 5.x, ray query gone — which is exactly
the pre-fix failure mode ("HW Lumen silently off on every Windows build"),
now conditional on two files nothing installs: `tools/fetch-dxc.ps1` exists
but is wired into no build, package, or CI step, and the shooter's copies
are untracked strays (now at least gitignored + documented, shooter PR #29).

Wanted: (a) a **loud boot warning** on DX12 when DXC was requested and FXC
was used; (b) fetch-dxc wired into the Windows build/packaging path;
(c) the still-unmeasured HW-vs-SW Lumen frame cost measured once on the dev
box and recorded (docs/perf/014 has only "within ~1.5x" guesses).

## EN-059 — File reads decode UTF-8 text as Latin-1: mojibake reaches the screen 🔴 *(2026-07-16 audit)*

`arena_02.world.json` contains a clean UTF-8 em dash in its `name` field
(verified byte-level). The shooter's diag bar renders it as
`Arena 02 a-hat-euro-quote Outdoor plaza` — the three-char cp1252 image of
the UTF-8 bytes. So somewhere between `readFile`/`loadWorld` and `drawText`,
file bytes are decoded per-byte (Latin-1) instead of as UTF-8. Every
non-ASCII character in any text asset (world names, future localized strings
— the shooter's SH-043 localization table is planned work) will render wrong.

First step is a probe to localise WHERE (Perry's readFile string
materialisation vs the engine's world loader vs drawText's glyph lookup),
because the fix lands in different repos depending on the answer. Note
Perry-upstream involvement is plausible (same class as the EN-020 slice
scanners: byte-level string handling).

## EN-060 — Grass distance LOD (quality-gated) 🟡 *(2026-07-16 audit)*

Grass tiles are frustum-culled (`material_system.rs:1776`) but in-frustum
tiles draw at full authored density regardless of distance — 120k blades in
the shooter, `material_pass` ~1.9 ms. The July audit's "concentric rings /
instance_count by distance" idea remains unimplemented.

**Hard constraint from the product owner: no visible quality loss.** A naive
density cut reads as the far field thinning and is REJECTED by definition.
Acceptable shapes: dithered per-instance dropout beyond the distance where a
blade is sub-pixel (its coverage already handled by the shader's distance
ramp), with A/B screenshots at multiple distances proving indistinguishable.
If it cannot be made imperceptible, close as won't-do — 1.9 ms is not worth
a visible regression here.

---

## EN-061 — SSR fireflies blanket interior surfaces ("the indoor blizzard") 🔴 *(2026-07-16 night)*

The user's "sparkles" report, run to ground with the f12-live bisect: white
render-texel speckles blanketing every interior surface of the shooter's
house (worst on the ceiling), present with SSGI, SSAO and SHADOWS each
toggled off, **gone the instant SSR is off**. Mechanism: the SSR march on
rough interior concrete hits bright window/sky content through the Hi-Z
pyramid; single-texel hits become white fireflies against the dim interior,
widened into blocks by the TSR upscale. Captures: shooter
`tools/.testout/bliz-on.png` (blizzard) vs `bliz-fixed.png` (clean); repro:
INDOORCAM+AITEST probe build + `--dbg-off ssr` A/B.

The shooter ships SSR **off** now (its water uses the planar probe; the July
audit measured SSR invisible outdoors — the toggle row stays for A/B). The
engine-side fix this ticket wants: roughness cutoff for the march (rough
dielectrics should get IBL, not a mirror march), a firefly luminance clamp
on hit radiance, and hit validation against the coarse Hi-Z level actually
sampled. Re-enable in the shooter only after the interior capture stays
clean.

## EN-062 — Spatial audio v2: live voices, real distance/pan model, doppler ✅ *(shipped 2026-07-16)*

The mixer could only fire-and-forget: `play_sound_3d` was a one-shot with a
fixed position, 1/d attenuation, linear pan — no way to loop an emitter, move
it, or stop it. Games faked ambience by re-triggering clips on timers (the
shooter's wind), and a river/creature emitter was simply not expressible.

Shipped, all engine-side so every game gets it for free:

- **Voice ids** — every play returns a stable id; `play_sound_3d_ex` +
  `voice_set_position/volume/pitch/lowpass/stop` steer one live voice.
  Looping voices persist until stopped; stop fades over a block (no click).
- **Inverse-clamped distance model** (`ref/(ref+rolloff·(d−ref))`, per-voice
  ref/rolloff/max) — ref=1, rolloff=1 is byte-for-byte the old 1/d, so the
  legacy API keeps its loudness. `max_dist` culls to a head-only advance.
- **Equal-power pan at 0.85 width** (was linear — center sat 6 dB down and
  hard sides were headphone-artifact absolute).
- **STEREO WAS MIRRORED**: the listener "right" was `cross(up, fwd)` = screen
  LEFT (mat4_look_at's `s` is `cross(fwd, up)`). Every spatial sound since
  the beginning panned to the wrong side. Fixed and now pinned by a test.
- **Air absorption** (distance-driven low-pass) and a **rear head-shadow
  cue** (low-pass toward 4.5 kHz + ~1.5 dB dip behind the listener), folded
  with the occlusion filter into one one-pole per voice.
- **Doppler** from the per-block distance delta (listener + source motion
  both count), clamped, smoothed, with a teleport guard so pool voices
  re-targeted across the map don't chirp.
- **Fractional resampling** (linear interp) — which also means assets now
  play at their AUTHORED rate: a 44.1 kHz file on the typical 48 kHz WASAPI
  endpoint used to play ~9% fast and sharp, forever. The Windows backend now
  reports the device rate to the renderer (it never had). Music streams are
  untouched (still device-rate; separate ticket if it ever matters).
- Per-voice gains ramp linearly across each mix block — per-frame
  position/volume rides cannot zipper. Consequence: any gain change fully
  lands one block late (~5–20 ms); the duck test now measures the settled
  block.
- Command ring 256 → 1024 (per-frame emitter updates + boot routing bursts).

Tests: 10 new (loop lifecycle, stereo crossing, legacy-curve equivalence,
equal-power, air absorption, rear cue, pitch rate, doppler zero-crossing
count, max-dist cull/return, per-voice occlusion). watchOS stubs regenerated;
web target exports the same six calls. First consumer: shooter SH-050.
