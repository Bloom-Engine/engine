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

## EN-003 — SSAO intensity / radius knobs 🟢

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

## EN-009 — Multi-mesh `drawMeshWithMaterial` ergonomics 🟢

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

## EN-012 — Foliage shading model in PBR ABI 🟡

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

## EN-015 — Imposter / billboard system 🟡

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

## EN-025 — Ragdoll FFI 🟡

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

## EN-026 — Particle / VFX system 🟡

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

## EN-027 — Deferred decal system 🟡

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

## EN-028 — Animation blending, masks, root motion 🟡

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

## EN-029 — Audio buses, reverb send, occlusion filter 🟡

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

## EN-030 — UI widget layer 🟢 *(TS-side, no Rust needed)*

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

## EN-031 — Gamepad backend verification + rumble 🟢

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

## EN-033 — Bone-socket world-transform query 🟢

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

## EN-038 — `bloom_take_screenshot` never fires on Windows 🔴

**Symptom.** `takeScreenshot('x.png')` from TS produces nothing: no file, and —
decisively — not even the unconditional `eprintln!("bloom: screenshot requested
-> '{}'")` at the top of the FFI, nor the `screenshot readback branch running`
log in `end_frame`. So the native function is never entered at all; this is not
a path/permissions problem or a failed readback.

**Repro (2026-07-12, shooter @ AAA round 1).** Call `takeScreenshot` from the
game loop at a fixed frame. A `console.log` immediately before it prints; the
engine's own log line never does. Three separate frames, plain relative path,
same result.

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

## EN-039 — immediate draw with a full transform 🟢

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

## EN-042 — Dynamic shadow-caster budget is 64, and overflow is silently dropped 🔴

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
