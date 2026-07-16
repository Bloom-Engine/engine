# RFC 0001 — Material system & render graph

**Status:** Implemented — Phases 1–8 + 10 shipped (ABI v3); Phase 3's pool is built but not wired (still Phase 3b); Phase 9 (water) is game-side. See §10 for as-built divergences.
**Author:** Ralph + Claude (pair)
**Target version:** 0.x → 0.(x+1), no public breakage for games on the existing
`drawModel` / `drawCube` / `loadModel` / `loadModelAnimation` APIs.

## Summary

Bloom's 3D renderer today has three separate shader pipelines (`pipeline_3d`,
`scene_pipeline`, a loaded-but-never-invoked `custom_pipelines`) each with its
own ad-hoc bind-group layout, and `end_frame_with_scene` is a ~500-line hand-
encoded sequence of passes. This shape makes it impossible to ship a custom
translucent material (water, glass, fog volumes, force fields, heat shimmer)
without engine patches every time.

This RFC proposes a small, coherent replacement:

1. A **shader ABI** — one documented bind-group layout and vertex format that
   every 3D shader (built-in PBR, custom user material) targets.
2. A **render graph** — `end_frame_with_scene`'s pass sequence becomes a
   declarative graph of nodes with explicit read/write contracts.
3. A **transient resource pool** — passes declare intermediate textures; the
   pool handles sizing, aliasing, and swapchain resize.
4. A **material contract** — materials declare opacity bucket (opaque /
   transparent / refractive) and which scene inputs they read; the graph
   inserts snapshot copies and sort keys automatically.
5. A **shader authoring pipeline** — WGSL in files, shared headers, hot
   reload in debug builds.

Water is the first consumer that exercises the whole stack. Glass is the
second, planned to prove the design generalises.

## Motivation

The concrete triggers are documented in the shooter repo:

- `shooter/docs/engine-fix-multi-skin.md` — a water-unrelated one-off patch
  the engine needed before the shooter could render multiple skinned
  characters.
- Conversations in the shooter's development log culminating in "what engine
  work do we need for real water?" — the answer was a bolt-on custom-shader
  draw call that solves 1/N of a much larger problem.

The broader reason: every time a game needs a material the built-in PBR
shader can't express, we either patch the engine or hack around it in game
code. Both cost days and leave debt. The industry-standard solution
(material graphs over a stable render-graph / shader-ABI substrate) is
well-understood — UE5's RDG, Frostbite's FrameGraph paper, Godot's render
server, Bevy's render graph. Bloom can adopt a minimal version of this
without adopting the kitchen-sink complexity.

The forcing function we'll use to validate the design is **water**. Water
needs every non-trivial capability — vertex displacement, scrolling normal
maps, scene-colour refraction, scene-depth shoreline fade, Fresnel
reflection from the env cubemap, interaction via an impulse texture. If
water falls out of the design as "a translucent material with scene-colour
input", the design is correct. If water needs engine special cases, it
isn't.

## Non-goals

- **No complex material graph editor.** Materials are WGSL source plus a
  tiny JSON descriptor. Visual graph authoring is a separate problem.
- **No deferred shading.** The engine is forward-rendered today and stays
  forward-rendered. Translucents already need a forward path; going
  deferred would be a separate RFC.
- **No shader code generation.** We rely on `naga-oil` (or a trivial text
  `#include` preprocessor) for sharing snippets between shaders. No custom
  DSL.
- **No automatic LOD for materials.** If a material is expensive it's the
  material author's problem.
- **No guarantee of binary-compatible migration across phases.** Each phase
  is a minor-version bump on the engine; games rebuild against the new
  engine. Public TS API stays source-compatible.

## Glossary

| Term           | Meaning                                                                   |
|----------------|---------------------------------------------------------------------------|
| Shader ABI     | The bind-group + vertex attribute layout contract every 3D shader targets |
| Pass           | One `wgpu::RenderPass` with defined colour/depth attachments              |
| Render graph   | Declarative description of all passes in a frame, their reads & writes    |
| Transient      | A texture that exists for part of a frame, managed by the pool            |
| Material       | A compiled shader pipeline + PBR/custom parameters + opacity bucket       |
| Bucket         | A rendering order class: `Opaque`, `Transparent`, `Refractive`, `Additive`|
| Scene input    | A render target from an earlier pass read by a later one                  |

---

## 1. Shader ABI

Every 3D shader compiled by the engine — whether it's the built-in PBR
shader, the sky shader, or a user-authored water material — binds the same
five groups in the same order. A shader that doesn't need a given group
omits its declaration; the pipeline layout still allocates the slot so
draws can batch cleanly.

### 1.1 Bind group table

| Group | Usage                | Stage(s)       | Stability                  |
|-------|----------------------|----------------|----------------------------|
| 0     | `PerFrame`           | Vertex + Frag  | One per frame              |
| 1     | `PerView`            | Vertex + Frag  | One per pass with a camera |
| 2     | `PerMaterial`        | Vertex + Frag  | One per material           |
| 3     | `PerDraw`            | Vertex + Frag  | One per draw call          |
| 4     | `SceneInputs` (opt)  | Fragment       | One per pass               |

Groups 0, 1, and 3 are **required** — every 3D shader binds them. Group 2
is required for PBR shaders and optional for custom shaders that don't
sample PBR textures. Group 4 is **opt-in** — only materials that declare
`readsScene = true` in their descriptor receive it.

### 1.2 `PerFrame` — group 0

Written once per frame, read by every 3D draw that frame. Layout (WGSL):

```wgsl
struct PerFrame {
  time:              f32,   // seconds since process start
  delta_time:        f32,   // seconds since last frame
  frame_index:       u32,   // monotonically increasing
  _pad0:             u32,
  screen_resolution: vec2<f32>, // physical pixels
  render_resolution: vec2<f32>, // render-target pixels (may be < screen w/ TSR)
  taa_jitter:        vec2<f32>, // sub-pixel offset applied to projection
  _pad1:             vec2<f32>,
};

@group(0) @binding(0) var<uniform> frame: PerFrame;
```

Rust mirror lives in `renderer/types.rs` as `PerFrameUniforms`, marked
`#[repr(C)]` and `bytemuck::Pod`.

### 1.3 `PerView` — group 1

Written at the start of each pass that has a camera (main HDR, shadow
cascades, water). Contains view/projection, lighting, env maps, shadow
data. Layout:

```wgsl
struct PerView {
  view:          mat4x4<f32>,
  proj:          mat4x4<f32>,
  view_proj:     mat4x4<f32>,
  prev_view_proj: mat4x4<f32>,  // for motion vectors
  inv_proj:      mat4x4<f32>,
  camera_pos:    vec4<f32>,    // xyz=pos, w=env_intensity
  camera_dir:    vec4<f32>,    // xyz=forward, w=fovy_rad

  ambient:       vec4<f32>,    // rgb + intensity
  fog:           vec4<f32>,    // r,g,b, density
  sun_dir:       vec4<f32>,    // xyz + intensity
  sun_color:     vec4<f32>,    // rgb + 0

  dir_light_count:   vec4<f32>, // x = count, yzw unused
  dir_lights:        array<DirLight,  4>,
  point_light_count: vec4<f32>,
  point_lights:     array<PointLight, 16>,

  shadow_splits:    vec4<f32>,
  shadow_view:      mat4x4<f32>,
  shadow_cascades:  array<mat4x4<f32>, 3>,
};

@group(1) @binding(0) var<uniform> view: PerView;
@group(1) @binding(1) var env_tex:         texture_2d<f32>; // equirect HDR
@group(1) @binding(2) var env_samp:        sampler;
@group(1) @binding(3) var env_diffuse_tex: texture_2d<f32>; // pre-filtered diffuse
@group(1) @binding(4) var brdf_lut_tex:    texture_2d<f32>;
@group(1) @binding(5) var brdf_lut_samp:   sampler;
@group(1) @binding(6) var shadow_tex_0:    texture_depth_2d;
@group(1) @binding(7) var shadow_tex_1:    texture_depth_2d;
@group(1) @binding(8) var shadow_tex_2:    texture_depth_2d;
@group(1) @binding(9) var shadow_samp:     sampler_comparison;
```

This is essentially the current `LightingUniforms` + env + shadow bindings
already in `scene_pipeline`, just moved to the new slot and enriched with
view/proj so a shader doesn't have to read them from two different groups.

### 1.4 `PerMaterial` — group 2

PBR material inputs follow the glTF 2.0 convention. Custom materials may
use only the `user_params` sub-binding.

```wgsl
struct MaterialFactors {
  metal_rough:  vec4<f32>,   // x=metallic, y=roughness, z=has_mr_tex, w=alpha_cutoff
  emissive:     vec4<f32>,   // rgb + 0
  base_color:   vec4<f32>,   // tint multiplier, rgba
  _reserved:    vec4<f32>,
};

@group(2) @binding(0)  var base_color_tex:   texture_2d<f32>;
@group(2) @binding(1)  var base_color_samp:  sampler;
@group(2) @binding(2)  var normal_tex:       texture_2d<f32>;
@group(2) @binding(3)  var normal_samp:      sampler;
@group(2) @binding(4)  var mr_tex:           texture_2d<f32>;
@group(2) @binding(5)  var mr_samp:          sampler;
@group(2) @binding(6)  var em_tex:           texture_2d<f32>;
@group(2) @binding(7)  var em_samp:          sampler;
@group(2) @binding(8)  var occ_tex:          texture_2d<f32>;
@group(2) @binding(9)  var occ_samp:         sampler;
@group(2) @binding(10) var<uniform> material: MaterialFactors;
@group(2) @binding(11) var<uniform> user_params: UserMaterialParams; // shader-defined
```

`UserMaterialParams` is declared by the shader, up to 256 bytes. Engine
uploads whatever the game passes through `drawMeshWithShader(uniforms: …)`.
White 1×1 textures are the defaults for all texture bindings a material
doesn't provide — this is what `scene_pipeline` does today; we codify it.

### 1.5 `PerDraw` — group 3

One per draw call. Contains transform + skinning.

```wgsl
struct PerDraw {
  mvp:          mat4x4<f32>,
  model:        mat4x4<f32>,
  prev_mvp:     mat4x4<f32>,  // for motion vectors
  model_tint:   vec4<f32>,
  skin_info:    vec4<u32>,    // x=joint_offset, y=joint_count, zw=unused
};

struct JointMatrices {
  matrices: array<mat4x4<f32>, 1024>,  // global joint buffer
};

@group(3) @binding(0) var<uniform>      draw:   PerDraw;
@group(3) @binding(1) var<uniform>      joints: JointMatrices;
```

`skin_info.x` gives the per-draw offset into the global joint buffer. This
is a direct continuation of the multi-skin scheme from
`shooter/docs/engine-fix-multi-skin.md`, now formalised.

### 1.6 `SceneInputs` — group 4 (opt-in)

Materials that declare `readsScene = true` in their descriptor receive this
group. Pipeline layout is created with or without it based on the shader's
reflected binding usage.

```wgsl
@group(4) @binding(0) var scene_color_tex:  texture_2d<f32>;   // HDR, pre-transparent
@group(4) @binding(1) var scene_color_samp: sampler;
@group(4) @binding(2) var scene_depth_tex:  texture_depth_2d;  // linearised in shader
@group(4) @binding(3) var scene_depth_samp: sampler;
@group(4) @binding(4) var impulse_tex:      texture_2d<f32>;   // world-space decals
@group(4) @binding(5) var impulse_samp:     sampler;
@group(4) @binding(6) var motion_vectors:   texture_2d<f32>;
```

The graph scheduler is responsible for inserting the snapshot copy that
produces `scene_color_tex` (via `copy_texture_to_texture`) and for binding
the linearised depth target. See §2.

### 1.7 Vertex attribute layout

Identical to today's `Vertex3D` (see `renderer/types.rs` line 171). Does
not change in this RFC.

```wgsl
struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal:   vec3<f32>,
  @location(2) color:    vec4<f32>,
  @location(3) uv:       vec2<f32>,
  @location(4) joints:   vec4<f32>,
  @location(5) weights:  vec4<f32>,
  @location(6) tangent:  vec4<f32>,
};
```

### 1.8 Fragment output profiles

Shaders declare one of two fragment output profiles.

**Profile `opaque`** — writes the full forward G-buffer. Matches today's
`fs_main_3d` / `fs_main_scene`:

```wgsl
struct OpaqueOut {
  @location(0) hdr:      vec4<f32>,
  @location(1) material: vec2<f32>,  // metallic, roughness
  @location(2) velocity: vec2<f32>,
  @location(3) albedo:   vec4<f32>,  // for SSGI
};
```

**Profile `translucent`** — writes HDR only, alpha-blended, no G-buffer
output. Used by water, glass, force fields, particles.

```wgsl
struct TranslucentOut {
  @location(0) hdr: vec4<f32>,
};
```

The material descriptor (§3) specifies which profile the shader uses. The
graph routes translucent draws to passes with single-target attachments.

### 1.9 Shared header

A single header WGSL file (`shared/shaders/material_abi.wgsl`) defines all
structs, bindings, and standard helpers (`world_from_skinned_position`,
`apply_tint`, `compute_motion_vector`, BRDF functions, shadow sampling,
fog). Every shader `#include`s it. Version evolution of the ABI = version
of the header.

### 1.10 Versioning

The header file carries `// ABI-VERSION: N` as a literal comment. Engine
parses this on pipeline creation; mismatch → hard error with migration
pointer. Version bumps require a migration note in this RFC and a chapter
in `docs/migration/abi-vN-to-vN+1.md`.

---

## 2. Render graph

`end_frame_with_scene` becomes a pure driver for the graph: build the
nodes, schedule them, execute. The actual pass bodies move into node
definitions.

### 2.1 Node shape

```rust
pub struct PassNode {
    pub name: &'static str,
    pub reads:  SmallVec<[PassInput;  4]>,
    pub writes: SmallVec<[PassOutput; 4]>,
    /// Optional ordering hints. Scheduler first honours data dependencies
    /// (a pass reading X runs after a pass writing X), then these.
    pub after:  &'static [&'static str],
    pub before: &'static [&'static str],
    pub run:    Box<dyn FnOnce(&mut Renderer, &mut NodeContext) + Send>,
}

pub enum PassInput {
    SceneColor,           // pre-transparent HDR snapshot
    SceneDepth,           // linearised scene depth
    Shadow(u8),           // cascade index 0..2
    EnvCubemap,
    MotionVectors,
    Impulse,
    Transient(TransientId),
}

pub enum PassOutput {
    HdrColor,             // main HDR RT (shared write target for opaque + water)
    Depth,
    MaterialRt,
    VelocityRt,
    AlbedoRt,
    Shadow(u8),
    Transient(TransientId),
    Swapchain,            // the surface texture — terminal
}
```

### 2.2 Scheduler

Trivial first version (sufficient for our ~15 passes):

1. Topologically sort by `reads` → `writes` edges.
2. Apply `after` / `before` tie-breakers.
3. Walk the sorted list; for each input that is `SceneColor` or
   `SceneDepth`, emit a synthetic `CopyToSample` node before the consumer
   if one isn't already present in the current segment.
4. Group contiguous nodes with identical colour/depth attachments into
   single `wgpu::RenderPass` calls (merge adjacent opaque draws, etc.).

This is O(N²) in pass count; fine for N<32. Smarter aliasing comes later.

### 2.3 Concrete frame graph (after migration)

```
shadow_0  shadow_1  shadow_2                [depth only, skinned + static]
             │
             ▼
         main_hdr   ┌── reads: Shadow(0..2), EnvCubemap
             │      ├── writes: HdrColor, Depth, MaterialRt, VelocityRt, AlbedoRt
             │      └── draws: sky, opaque static, opaque skinned, scene-graph
             │
             ▼
         ssao       ┌── reads: Depth, AlbedoRt  → writes: Transient(Ssao)
             │
             ▼
         ssgi       ┌── reads: Depth, AlbedoRt, prev SceneColor
             │      └── writes: Transient(Ssgi)
             │
             ▼
         ssr        ┌── reads: Depth, SceneColor, MaterialRt  → writes: Transient(Ssr)
             │
  [synthetic CopyToSample: HdrColor → SceneColor]
             │
             ▼
         translucent_refractive  ┌── reads: SceneColor, SceneDepth, EnvCubemap
             │                   └── writes: HdrColor (alpha blend)
             │
             ▼
         translucent_additive    ┌── writes: HdrColor
             │
             ▼
         taa        ┌── reads: HdrColor, prev, Velocity  → writes: Transient(HdrTaa)
             │
             ▼
         bloom_ladder (downsample × N, upsample × N)
             │
             ▼
         scene_compose  ┌── reads: HdrTaa, Transient(Bloom*)  → writes: ComposedRt
             │
             ▼
         tonemap → hud → Swapchain
```

The water pass is the `translucent_refractive` group — no special case.
Glass, force fields, fog volumes slot into the same node as additional
draws.

### 2.4 Transient resources

```rust
pub struct TransientDesc {
    pub id:     TransientId,
    pub format: wgpu::TextureFormat,
    pub size:   SizePolicy,          // Swapchain, HalfSwapchain, QuarterSwapchain, Fixed(w, h)
    pub usage:  wgpu::TextureUsages,
}
```

Pool reference-counts; when a transient is last-read, its backing texture
returns to the pool and can be aliased by a later pass with compatible
format/size. Resize events reset the pool entirely.

---

## 3. Material contract

### 3.1 Descriptor

JSON or programmatic, loaded alongside a `.wgsl` file:

```json
{
  "name": "water_river",
  "shader": "materials/water.wgsl",
  "bucket": "refractive",
  "reads_scene": true,
  "writes_profile": "translucent",
  "parameters": {
    "flow_dir":     [1.0, 0.0],
    "wave_amp":     0.15,
    "foam_threshold": 0.8,
    "scroll_speed_a": 0.02,
    "scroll_speed_b": 0.037,
    "absorption":   [0.4, 0.7, 0.9]
  },
  "textures": {
    "base_color_tex": "materials/water_color.png",
    "normal_tex":     "materials/water_normal.png"
  }
}
```

### 3.2 Buckets and sort keys

```rust
pub enum Bucket {
    Opaque,                                   // sort front-to-back by depth (early-z)
    Transparent,                              // sort back-to-front by depth
    Refractive,                               // same as Transparent, reads SceneColor
    Additive,                                 // any order; no sort
}
```

Draw submission stores `(bucket, -z_camera_space, material_id)`. Graph
splits draws across pass nodes by bucket. Within a bucket, sort key breaks
ties deterministically (same frame, same order — important for TAA).

### 3.3 Material instance

A game creates a material from a descriptor:

```ts
const waterMat = loadMaterial("materials/water_river.json");
drawMeshWithMaterial(waterMat, waterMesh, position, scale, tint);
```

`drawMeshWithMaterial` is the same draw call as `drawModel`, just with an
explicit material id. Internally, the existing `drawModel` is a shorthand
for "use the built-in PBR material".

---

## 4. Shader authoring

### 4.1 File layout

```
engine/shared/shaders/
  material_abi.wgsl          ← header, included by all
  common/
    pbr.wgsl                 ← BRDF, IBL sampling
    shadows.wgsl             ← PCF, cascades
    fog.wgsl
    tonemap.wgsl
    sky.wgsl
  materials/
    pbr_opaque.wgsl          ← default built-in (replaces scene_pipeline shader)
    pbr_unlit.wgsl
    sky.wgsl                 ← moves out of the renderer's string constants
    water.wgsl               ← first translucent
    glass.wgsl
```

Shaders are loaded via `include_str!` at engine build time. The
preprocessor resolves `#include "common/pbr.wgsl"` by simple text
concatenation (naga-oil is the standard crate; we can adopt it or write ~80
lines of equivalent). Duplicate-include guards are by `#ifndef` sentinel
names in each header.

### 4.2 Hot reload (debug builds)

On `cfg(debug_assertions)`, the engine spawns a `notify` watcher over
`shared/shaders/`. A file change triggers:

1. Re-read the file(s).
2. Re-run the preprocessor.
3. `wgpu` pipeline creation — fast enough on M1 (<100 ms per pipeline).
4. Atomic swap the new `RenderPipeline` into place.
5. Game code keeps the same `ShaderHandle`.

If compilation fails, log the error and keep the old pipeline. No crash.

### 4.3 Release builds

Release builds bake the concatenated WGSL strings with `include_str!` at
compile time and never touch the filesystem. No watcher, no runtime
preprocessing. This is the current model; we keep it.

---

## 5. Migration plan

Each phase is a mergeable increment. The shooter builds against every
merged main; a phase that would regress visuals in the shooter is rejected.

### Phase 0 — This RFC lands

Nothing implemented. Just this doc and the updated phase checklist
committed to `main`.

**Acceptance:** RFC reviewed and merged.

### Phase 1 — Shader ABI header ✅

- [x] Write `shared/shaders/material_abi.wgsl` with the definitions in §1.
- [x] Write `shared/shaders/common/{pbr,shadows,fog,tonemap,sky}.wgsl`.
- [x] Refactor the three existing 3D shaders to include the
      header and use group 0-3.
- [x] Rust-side: refactor `LightingUniforms`, `Uniforms3D`, etc. so
      `PerFrame`, `PerView`, `PerMaterial`, `PerDraw` are distinct types
      mirroring the WGSL.
- [x] Pipeline creation (`pipeline_3d`, `scene_pipeline`) switches to new
      layouts. `custom_pipelines` path removed (dead code today anyway).

(As-built: the ABI is at **version 3** — see §10 for what grew past this
spec. The version check lives in `renderer/shader_include.rs` +
`renderer/shader_library.rs`, not the long-gone `shaders.rs`.)

**Acceptance:**
- Shooter SELFTEST screenshots (stable seed): identical within 1-bit
  tolerance before and after.
- Perry compilation succeeds; shooter runs; all features unchanged.
- No public API changes.

### Phase 2 — Render graph skeleton ✅

- [x] `renderer/graph.rs`: `PassNode`, `PassInput`, `PassOutput`,
      scheduler, executor.
- [x] Port the existing passes of `end_frame_with_scene` one by one to
      `PassNode`s. The function becomes `build_graph() + graph.execute()`.
- [x] Nothing else changes. Pass bodies are moved verbatim into
      `run:` closures.

**Acceptance:** same SELFTEST tolerance, same timings within 2%.

(As-built: the graph drives the real frame — `mod.rs` marks "RFC 0001
Phase 2b — complete". Node names and the scheduler shape differ from §2 —
see §10.)

### Phase 3 — Transient resource pool 🟡 built, not wired

- [x] `renderer/transient.rs`: `TransientDesc`, pool with reference
      counting (aliasing was dropped — reuse only).
- [ ] Ssao, ssgi, ssr, bloom mip chain, TAA history → all declared as
      transients.
- [ ] Resize handling goes through the pool; the renderer's member fields
      shrink.

**Acceptance:** GPU memory usage reduced (target: 20%+ on 1080p scene),
SELFTEST identical. **Not yet met** — the pool exists with unit tests, but
in the live graph `Transient(u32)` ids are ordering tokens over persistent
renderer fields ("Phase 3b" per `transient.rs`).

### Phase 4 — Translucency buckets + scene snapshot

- [ ] Draw submission API grows a `Bucket` enum. Existing `drawModel`
      remains `Bucket::Opaque` by default.
- [ ] Graph gains `SceneColor` as an input kind. Scheduler synthesises
      `CopyToSample` nodes.
- [ ] Translucent and refractive pass nodes added after opaque.

**Acceptance:** still no behaviour change for existing content (no game
currently submits non-opaque draws). Synthetic unit test: two
alpha-blended cubes render in back-to-front order regardless of
submission order.

### Phase 4c — Scene depth snapshot ✅

- [x] Opaque depth texture gains `COPY_SRC` usage.
- [x] Translucent-pass path acquires a transient `Depth32Float` in
      parallel with the scene-colour transient whenever a material
      declares `reads_scene = true`, copies the live depth into it,
      and binds it at `group(4) binding(2)` — the live depth view is
      still the pass's own depth-stencil attachment, so wgpu no
      longer sees a sampled-and-attached aliasing conflict.
- [x] `update_scene_inputs` takes `Option<&TextureView>` for depth and
      falls back to the internal 1×1 stub when no translucent draws
      need it.

**Acceptance:** water shoreline fade in `shooter/src/main.ts` reads
`scene_depth_tex`, linearises via `view.inv_proj`, and produces a soft
dry-to-wet transition at the river edges. No regressions in the
opaque material path (matTest cube retired anyway in Phase 9).

### Phase 5 — Material params UBO ✅

- [x] `PerMaterial` group `user_params` binding (`@group(2)
      @binding(11)`) — already declared by the ABI; this phase wired
      it through `MaterialSystem` so each handle gets its own per-
      material BindGroup once `set_user_params` is called.
- [x] `setMaterialParams(handle, params: number[])` TS API + FFI
      (`bloom_set_material_params(handle, ptr, count)`). 256-byte cap
      enforced; 64 f32 slots. Pass an empty array to revert to the
      default zero-initialised UBO. (Chose this shape over a
      `userParams[]` arg on `drawMeshWithMaterial` because the
      params are per-material, not per-draw — uploading 60×/sec when
      they only change on tuning would be wasteful.)
- [ ] Material descriptor JSON loader (`loadMaterial(path)`).
      *Deferred* — when shooter materials get pulled out of inline
      TS strings into disk WGSL files (concurrent with Phase 6 hot
      reload), a JSON loader fits as a thin wrapper around
      `compileRefractiveMaterial(readFile(...))` +
      `setMaterialParams`.

**Acceptance:** ✅ shooter water material now reads its tint /
absorption mix / foam strength / rim brightness / sky LOD from a
`WaterParams` UBO bound at `@group(2) @binding(11)`. Verified by
swapping the tint from `(0.10, 0.30, 0.40)` to `(0.55, 0.10, 0.10)`
at runtime — water visibly turned red, no WGSL recompile.

### Phase 6 — WGSL file layout + hot reload ✅

- [x] Engine-internal shaders already live in
      `native/shared/shaders/*.wgsl` and are `include_str!`-baked
      via `shader_library.rs` (predates this phase).
- [x] `notify`-based watcher in `renderer::hot_reload` —
      `MaterialHotReload` owns a `RecommendedWatcher` that forwards
      Modify/Create events through an mpsc channel; the main thread
      drains them in `Renderer::poll_material_hot_reload` (called
      from `EngineState::end_frame`). 120 ms debounce coalesces the
      multi-event burst macOS fires per save.
- [x] `compile_material_from_file(path, profile, bucket, reads_scene)`
      path on `Renderer`, plus `bloom_compile_material_from_file`
      FFI and `compileMaterialFromFile(path, bucket)` TS helper.
      Recompile on file-change replaces the pipeline at the same
      MaterialHandle, so existing draws automatically pick up the
      new shader on the next frame.
- [x] Failures during reload (parse error, validation) are logged
      and the previous pipeline keeps running — never crashes the
      game.
- [ ] Release build has no watcher thread. *Skipped* — the `notify`
      worker is cheap (one OS-event-pump thread, idle most of the
      time) and games may want hot reload in non-dev builds for
      ad-hoc tuning. A `cfg(debug_assertions)` gate is a one-line
      change if a future ship-mode demands it.

**Acceptance:** ✅ shooter water material now lives at
`shooter/assets/materials/water.wgsl`, loaded via
`compileMaterialFromFile`. Editing the file while the game runs
fired `[hot_reload] reloaded "/.../water.wgsl" (handle 2)` and the
on-screen water visibly switched colour without a crash.

### Phase 7 — Impulse texture ✅

- [x] `R32Float` 256×256 world-space texture (ping-pong pair) in
      `renderer::impulse_field::ImpulseField`, persistent across
      frames rather than transient — splats need to decay for ~2 s,
      which straddles multiple frames.
- [x] `splat_impulse(world_pos, radius, strength)` gameplay API via
      `bloom_splat_impulse` FFI / `splatImpulse()` TS helper.
- [x] Decay + splat combined in one compute pass
      (`impulse_field.wgsl`, 8×8 workgroup): reads previous field,
      multiplies by per-frame decay (~0.968 for a 2 s half-life at
      60 fps), then adds every queued splat with quadratic falloff.
- [x] Bound at `@group(4) @binding(4)` in scene_inputs whenever a
      translucent material declares `reads_scene` — the scene_inputs
      layout switched binding 4 to `Float { filterable: false }` and
      binding 5 to `NonFiltering` sampler (R32Float is non-filterable
      without a feature flag).

**Acceptance:** ✅ shooter water shader samples `impulse_tex` via
`textureLoad`; walking across the river submits a splat per frame
and leaves a visible, decaying trail. Diagnostic red-channel
visualisation showed a clean path of accumulated impulses along the
player's route.

### Phase 8 — Observability ✅

- [x] Per-pass GPU timer queries (already wired into wgpu
      `TimestampWrites` on every render/compute pass; profiler folds
      them into a 120-frame rolling window).
- [x] `Profiler::snapshot()` + `bloom_profiler_overlay_text()` FFI +
      `getProfilerOverlay()` TS helper expose per-pass averages.
- [x] Shooter F3 toggles `setProfilerEnabled` + an overlay that lists
      all 28+ passes with CPU / GPU µs, sorted by CPU time descending.
- [x] Frame histogram over the last 120 frames.
      `Profiler::frame_history()` returns `(cpu_us, gpu_us)` pairs
      in chronological order; FFI
      `bloom_profiler_frame_history()` + TS
      `getProfilerFrameHistory()` expose them. Shooter F3 overlay
      renders bars above the per-pass table — bars over the 16.7
      ms (60 fps) reference line are drawn red, under-budget bars
      blue.

**Acceptance:** overlay renders every pass with stable sub-ms readings
on the shooter. Zero per-frame cost when disabled — `Profiler::begin`
early-returns on `!self.enabled` and no timestamp queries are
reserved, so the GPU cost is the unavoidable `ResolveQuerySet` of 0
pairs (no-op).

### Phase 9 — Water

- [ ] `engine/shared/shaders/materials/water.wgsl` implementing Gerstner
      waves, scrolling dual normal maps, Fresnel, SceneColor refraction,
      SceneDepth shoreline fade.
- [ ] `shooter/assets/materials/water_river.json` + `shooter/tools/build-
      props.ts` extended to emit `water_river.glb` (flat tessellated
      quad).
- [ ] `shooter/src/main.ts` swaps the tessellated-cube river loop for
      `drawMeshWithMaterial(waterMat, ...)`.

**Acceptance:** side-by-side screenshot review. Water reads as real
water at distance and close-up. Banks fade. No visible seams or aliasing.

### Phase 10 — Second consumer (glass) ✅

- [x] Glass WGSL (inline in `shooter/src/main.ts`) reusing
      `material_abi.wgsl` + `common/pbr.wgsl` and the SceneColor
      (group 4) infrastructure. Flat normal, Schlick Fresnel with
      F0=0.04, 0.008 refraction offset, faint cyan transmission tint.
- [x] A glass pane placed in the south-wall door opening of house h1
      at `(-21, 0, -10)` in the arena_02 world.

**Acceptance:** ✅ no engine change between phase 9 and 10. Glass
compiled via the same `compileRefractiveMaterial` entry point as water
and rendered correctly; the diag pane in front of the player spawn
showed the expected Fresnel-biased transmission + sky reflection.

---

## 6. Open questions

These are deliberately left open for reviewer input. Each will be
closed-out before the relevant phase begins.

1. **Depth linearisation.** Raw depth vs linearised depth as the bound
   `SceneDepth` texture? Raw is cheaper but every consumer shader has to
   linearise; linearised is a one-time cost and simpler downstream. Lean:
   linearised, written into a transient by a tiny copy shader during the
   `CopyToSample` synthesis.

2. **Scene colour format.** Full HDR `Rgba16Float` sample, or a
   tonemapped `Rgba8Unorm` copy? HDR is correct for refraction (bright
   lights stay bright through water); 8-bit is 2× cheaper. Lean: HDR.
   Revisit if mobile-target perf demands.

3. **Scheduler API shape.** Closures vs. explicit trait objects for
   `run`. Closures are simpler for migration; trait objects are easier to
   unit-test. Lean: closures now, trait refactor deferred.

4. **Impulse texture coordinate system.** World XZ projected top-down (good
   for water / snow / mud), or per-material UV (good for walls)? Lean:
   world XZ for the first version; wall decals are a later RFC.

5. **Material JSON vs TOML vs pure Rust registration.** JSON is the least
   friction for game authors; Rust registration is zero-runtime-cost and
   type-safe. Lean: JSON for game data, Rust for built-ins.

6. **ABI version bumping cadence.** Every field change = version bump,
   or batched? Lean: every change; versions are cheap and cheap to
   migrate.

7. **Per-view vs per-pass uniform writes.** Shadow cascades want three
   different `PerView`s; main HDR wants one. Implement `PerView` as a
   slotted UBO indexed by pass? Or as a dedicated transient? Lean:
   slotted (up to 8 views), avoids per-frame alloc.

---

## 7. Risks

- **Graph scheduler bugs.** An incorrectly ordered pass will manifest as
  subtle rendering glitches rather than crashes. Mitigation: every phase
  ships with SELFTEST screenshots, hash-compared in CI.
- **WGSL header bloat.** Including the 400-line PBR header in every shader
  slows pipeline compilation. Mitigation: profile it; at 100 ms per
  pipeline on M1 we can tolerate up to ~20 pipelines without perceptible
  startup cost.
- **Hot reload state leaks.** If a pipeline is swapped mid-frame we get a
  torn frame. Mitigation: swap happens at `begin_frame`, never inside a
  pass.
- **Perry codegen interactions.** TS-side material APIs traverse the usual
  FFI boundary. We've already catalogued three Perry quirks in
  `shooter/docs/perry-quirks.md`; new APIs must respect them (no throws in
  reachable control flow, no `{key, shorthand}` returns, no JSON.parse
  array lengths). Mitigation: every new FFI function reviewed against
  that doc.
- **Over-engineering.** A full graph framework could spiral. The scheduler
  is deliberately kept tiny (~100 LOC) and un-fancy. We add features only
  when a phase needs them.

---

## 8. Alternatives considered

- **Just add a `drawMeshWithShader` FFI.** Faster (4-6 h), solves water
  alone, leaves debt. See the earlier correspondence for the trade-off.
- **Adopt an existing render graph crate (rend3, Kajiya).** These are
  huge and opinionated; integrating would be more work than writing ~300
  lines ourselves.
- **Go deferred.** Modern engines go deferred for a reason, but the move
  is disruptive and orthogonal to material flexibility. Forward is fine
  at our scale.
- **Material graph editor.** Valuable long-term, out of scope here. If
  desired, the material JSON from §3.1 is a stable ingest format that a
  future editor can author.

---

## 9. Decision

Proceed with phases 1 through 10 in order. Each phase is a PR against
`main` referencing this RFC. The RFC itself becomes a living document —
amendments welcomed as we learn.

The shooter is the primary test harness. Water is the forcing function.

---

## 10. As-built divergences (audit 2026-07-16)

The design shipped, but the code moved past this spec in places. The
proposal sections above are left as written; this section is the map from
spec to reality. Ground truth: `native/shared/shaders/material_abi.wgsl`
(ABI **v3**), `renderer/graph.rs`, `renderer/transient.rs`.

**ABI (§1):**
- `PerView` light arrays grew: `array<DirLight, 8>` and
  `array<PointLight, 256>` (froxel-culled), not 4/16.
- `PerFrame` gained `wind: vec4<f32>` (EN-013) and `cloud: vec4<f32>`
  (EN-040) after `taa_jitter`.
- `MaterialFactors._reserved` became `shading_model: vec4<f32>` +
  `foliage_params: vec4<f32>` (EN-012).
- Group 2 extends past binding 11: binding 11 is left for the shader's own
  `user_params` declaration, bindings 12-13 are planar reflection (EN-011),
  14-17 are terrain texture arrays (EN-014).
- ABI version bumps: the `// ABI-VERSION: N` check works as described but
  lives in `renderer/shader_include.rs` (`abi_version_of`) +
  `renderer/shader_library.rs` (`verify_abi_version`). The promised
  `docs/migration/abi-vN-to-vN+1.md` files were never created; v1→v3
  changes are documented in the ABI header itself.

**Render graph (§2):**
- `PassNode` is generic `PassNode<'a, Ctx>` with `run: Box<dyn FnOnce(&mut
  Ctx) + 'a>` — no `Renderer`/`NodeContext` split, `Send` dropped,
  `PassInput::Transient(u32)` instead of a typed id.
- The scheduler is a plain Kahn topological sort with `after`/`before`
  tie-breaks. §2's step 3 (synthesized `CopyToSample` nodes) and step 4
  (merging contiguous passes into one `wgpu::RenderPass`) were never
  implemented; the live graph pins order with `with_after`.
- As-built node names: `froxel_assign`, `shadow` (one node, all 3
  cascades), `hdr_scene`, `pt`, `translucent`, `hiz_build`,
  `occlusion_capture`, `gtao`, `ssao_blur`, `ssr_march`, `ssr_temporal`,
  `ssgi`, `bloom`, `compose`, `postfx_tail`, `auto_exposure`, plus
  sub-graphs `material_pass` and `overlay_2d`. (§2.3's `main_hdr`/`ssao`/
  `taa`/`scene_compose`/`tonemap` names never existed as built.)
- `TransientDesc` has no `id` field (separate `TransientId` handle), adds
  `mips`/`samples`, and the pool ref-counts and reuses but does **not**
  alias.

**API (§3):**
- `loadMaterial` takes a `MaterialDesc` object (`{shader, bucket}`), not a
  JSON path (the JSON loader was deferred, as §5's note says).
Glass is the generality check.
