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

## EN-005 — Atmospheric scattering / sun disk 🔴

**Why:** the HDR environment provides a static sky panorama with
clouds, but no real sun disk + atmospheric scattering. The sky
reads as "very nice photograph" rather than "physically alive
sky" — fine outdoors, but breaks down at sunset / the moment the
sun moves.

**Scope:** large — a real sky shader (Rayleigh + Mie scattering,
parameterised by sun direction + density) replacing or
augmenting the existing sky pass.

**Sub-RFCs needed:**
- Should the sun-shaft pass tap the new sky's transmittance?
- Time-of-day API: do we want it, and if so, how does it interact
  with `setDirectionalLight`?

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
