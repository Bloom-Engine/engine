# Engine ticket backlog

Outstanding engine work surfaced while building the shooter game.
Game-side counterparts live in `bloom/shooter/docs/tickets.md`.

Status legend: 🟢 ready · 🟡 has open design questions · 🔴 needs
broader RFC

---

## EN-001 — Instanced-draw FFI 🟢

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

## EN-002 — `drawModel` rotation parameter 🟢

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

## EN-010 — Alpha-cutout bucket 🟢

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

## EN-011 — Planar reflection capture 🟢

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

## EN-013 — Global wind UBO in PerFrame 🟢

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

## EN-014 — Texture-array binding pattern for splat-mapped terrain 🟡

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

## EN-016 — Custom-material shadow-receive helper 🟢

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

## EN-017 — Post-pass slot for game-side full-screen FX 🟡

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

**Blocked on:** access to a Windows 11 box with a HiDPI display
(or a VM forwarding DPI), and a Linux dev with X11. Web checks
can be done from any modern browser on the macOS dev box.

## EN-020 — Native AV: heap read overrun, layout-sensitive 🔴

**Symptom.** Bloom Shooter round-2 audit (2026-07-04): three scripted runs
crashed with c0000005 at the same instruction (`main.exe+0xe8e5` on the
af98dbe-era link), reading `0x…FFF8` — 8 bytes below a page boundary.
No Rust panic output (engine builds `panic = "abort"`, which would print),
so this is raw UB: something reads past the end of a heap allocation and
faults only when the allocation abuts an unmapped page.

**What is ruled out.** Runtime feature toggles alone (a 19-transition
ssgi/shadows/profiler gauntlet under combat load survived); profiler
string churn alone (one crash happened with only ~8 small prints);
specific stages (three different ones). The fault did not reproduce after
a relink (two 60 s fishing runs) — consistent with layout sensitivity,
not with a fixed trigger.

**Standing kit.** WER LocalDumps armed on the dev box (dumps →
`shooter/tools/.testout/dumps/`, dialog suppressed), line tables in the
staticlib (`debug = "line-tables-only"`), `perry compile --debug-symbols`
emits `main.pdb`, LLVM symbolizer available. See
[crash-triage-windows.md](crash-triage-windows.md). Next occurrence =
symbolized stack; fix then.

**Suspect space.** Perry-runtime heap/string handling at the FFI boundary
(profiler overlay/history strings are the heaviest string traffic in the
crashing runs), or an engine-side heap read overrun in a small shared
helper. Audit report: `shooter/docs/audit-round2.md` finding F1.

## EN-021 — SSR + IBL specular exclusive ownership 🟡

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

## EN-022 — Motion vectors for material-system draws 🔴

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

## EN-023 — GI software path: colored bounce is unreachable 🟡

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

