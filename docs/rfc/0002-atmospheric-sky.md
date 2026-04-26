## RFC 0002 — Atmospheric scattering & sun disk (EN-005)

**Status:** Accepted — 2026-04-26
**Author:** Ralph + Claude (pair)
**Target version:** 0.x → 0.(x+1), additive API. The existing
static HDR environment path stays available for users who want to ship
a panorama; the procedural sky is opt-in.

## Summary

Replace the static HDR-panorama sky with a physically-based atmosphere:
Rayleigh + Mie scattering driven by sun direction, with a real sun disk
and altitude-aware aerial perspective. The model is Hillaire 2020 (the
"scalable atmosphere" 4-LUT approach used by UE5, Frostbite, Godot 4) —
two precomputed LUTs (transmittance, multi-scattering) and two
per-frame LUTs (sky-view, aerial-perspective).

Both open questions in EN-005 are answered:

1. **Sun-shafts tap the new sky's transmittance.** The shaft's tint is
   no longer user-set — it comes from `transmittance(sun_dir)`, which
   gives a free, physically-correct warm shaft at sunset.
2. **`setSunDirection()` becomes the source of truth.** It updates
   both the sky's sun input *and* the directional-light uniform
   (direction + a transmittance-derived color). `setDirectionalLight`
   stays as a lower-level escape hatch for games that want artistic
   control or no atmosphere.

## Motivation

The current sky (`shaders/common/sky.wgsl`, `render_sky_pass` at
`renderer/mod.rs:7025`) is a thin equirect sampler over a static
HDR panorama with a baked GGX mip chain for IBL. It reads as "very
nice photograph" but breaks the moment the sun moves: shadows
re-orient, the sky doesn't; the directional-light color is whatever
the user typed, untethered from any sky. A real sky shader closes
that gap and unlocks day/night, weather variation, and altitude
flight without re-baking panoramas per scenario.

## Non-goals

- Clouds (volumetric or otherwise). Hillaire 2020 explicitly composes
  with a separate cloud pass; we defer that to a follow-up ticket.
- Night sky / stars / moon. The sky goes dark below the horizon when
  the sun sets. Stars are a separate texture composite, deferrable.
- Weather (rain, fog density variation). The model exposes Rayleigh +
  Mie density scalars; weather can ride on those later, out of scope here.
- Replacing IBL prefilter. The procedural sky will *feed* the same
  prefilter chain (`load_env_from_hdr` flow); we don't redesign IBL.

## Architecture

Four LUTs, three render-time passes, one new TS API surface.

### LUTs

Sizes are tiered by target. Desktop (mac / windows / linux) gets
Hillaire's defaults; web (wasm32) and mobile (iOS / Android) get a
smaller tier to keep memory + per-frame bake cost down. Tier is
chosen at compile time via `cfg(target_arch)` + `cfg(target_os)`,
not runtime.

| LUT | Format | Desktop | Web / mobile | Lifetime | Inputs |
|---|---|---|---|---|---|
| Transmittance | `Rgba16Float` | 256×64 | 128×32 | Bake once at init | (μ = cosθ, height) |
| Multi-scattering | `Rgba16Float` | 32×32 | 32×32 | Bake once at init | (μ_s = cos sun-zenith, height) |
| Sky-view | `Rgba16Float` | 192×108 | 128×72 | Per-frame (sun moved) | sky hemisphere from camera |
| Aerial-perspective | 3D `Rgba16Float` | 32×32×32 | 16×16×16 | Per-frame (sun moved) | view-frustum-aligned scattering |

Transmittance + multi-scattering are baked once via a compute pass at
renderer init (mirrors the BRDF-LUT pattern at `renderer/mod.rs:6780`).
Sky-view + aerial-perspective are recomputed when the sun direction or
atmosphere parameters change — i.e., once per `setSunDirection()`, not
every frame for a static sun.

### Render passes

```
HDR render
  ├─ render_atmosphere_pass    ← NEW, replaces render_sky_pass when procedural
  │    samples sky-view LUT, draws sun disk
  ├─ opaque 3D pass
  │    aerial-perspective LUT injected into PBR shader fog term
  └─ scene_compose_pass
       sun-shafts now multiply by transmittance(sun_dir)  ← coupling answer
```

The procedural sky path is selected by a renderer flag set when
`setProceduralSky(...)` is called; otherwise the existing
`render_sky_pass` runs unchanged. No regression risk for games using
panoramas.

### IBL coupling

Procedural sky still needs to feed IBL (specular env + diffuse SH) so
materials light correctly under the same sky. On `setSunDirection()`,
after sky-view + aerial LUTs update, we:

1. Render the sky into a low-res cubemap (e.g. 64×64×6).
2. Run the existing GGX prefilter (Karis 2013) over it to populate
   the same mip chain `load_env_from_hdr` produces.
3. Material PBR shader is unchanged.

The cubemap re-bake is the dominant cost of `setSunDirection()` —
~0.3 ms at 64-cube + 5 mip levels, only when the sun moves. Static
suns pay it once.

## API surface

```typescript
// New procedural-sky API (opt-in)
export function setProceduralSky(opts?: {
  rayleighDensity?: number;   // default 1.0
  mieDensity?: number;        // default 1.0
  groundAlbedo?: Color;       // default soil-grey
}): void;

export function setSunDirection(dir: Vec3, intensity?: number): void;

// Existing APIs unchanged
export function setDirectionalLight(direction: Vec3, color: Color, intensity: number): void;
export function loadEnvironment(path: string): void;  // panorama path
```

### Sun direction → directional light coupling

When `setProceduralSky` is active, `setSunDirection(dir, intensity)`:

1. Updates the atmosphere uniform `sun_dir`.
2. Triggers sky-view + aerial-perspective + IBL re-bake.
3. **Writes `lighting_uniforms.light_dir = dir` and
   `light_color = transmittance(dir) * sun_radiance * intensity`.**
   This is the source-of-truth coupling: one call, sky and sunlight
   stay consistent.

When `setProceduralSky` is *not* active, `setSunDirection` is a no-op
(returns warning in debug). Games either pick the procedural path
(and use `setSunDirection`) or the panorama path (and use
`setDirectionalLight`). They can mix — calling `setDirectionalLight`
after `setSunDirection` overrides the auto-color, which is the escape
hatch for stylized lighting on top of a procedural sky.

### Time-of-day

Deliberately *not* in V1. `setSunDirection` is the primitive; a
time-of-day helper (`setTimeOfDay(hour, latitude)` → sun direction)
is a 20-line user-space utility we can ship in `src/scene/sky.ts`
once the underlying primitive is stable. Putting time-of-day in the
engine core would lock in calendar/latitude semantics that games may
want to override (alien planets, accelerated cycles, etc.).

## Sun-shafts

The current shaft pass (`shaders.rs:4505`) takes a user-set
`sun_shaft_color` vec3. Under procedural sky:

- `sun_shaft_color` is *computed* per-frame as
  `transmittance(sun_dir) * sun_radiance`, written into the existing
  `SceneComposeParams.sun_shaft_color` field.
- `setSunShaftColor()` still works as an override (same escape-hatch
  pattern as directional light).

This is the answer to sub-RFC #1: shafts tap transmittance, no API
break, free physical sunset warmth.

## Phasing

**Phase 1 — Bake-only LUTs (no rendering yet).**
Land transmittance + multi-scattering LUT compute passes, expose them
as renderer-internal textures. Headless test verifies LUT contents
against reference (Hillaire's published values). ~1–2 days.

**Phase 2 — Procedural sky pass.**
Sky-view LUT + `render_atmosphere_pass`. `setProceduralSky` /
`setSunDirection` TS APIs. No IBL coupling yet (sky lights itself,
materials still use whatever environment was last loaded).
~2–3 days.

**Phase 3 — IBL + sun-shaft coupling.**
Equirect re-bake on `setSunDirection` (chose equirect over cube to
reuse the existing GGX prefilter chain). Sun-shaft transmittance tap.
Aerial-perspective LUT *deferred to Phase 4* — the 3D LUT plus PBR
shader changes were a bigger surface area than the IBL/shaft work
warranted in one PR. ~1.5 days.

**Phase 4 — Sky-tinted fog + polish.**
CPU-derived fog color tracking the sun direction (analytic
horizon-tint blend, no GPU compute), plus zenith-banding dither in
the procedural sky shader. Sun disk + limb darkening already
landed in Phase 2. ~0.5 day.

The full 3D aerial-perspective LUT (32³ / 16³, per-frame compute,
PBR fog-term integration) is left as **EN-005 V2** — a follow-up
ticket. The Phase 4 sky-tinted fog captures the dominant signal
(warm haze at sunset, cool blue at noon) without the bind-group
plumbing or per-frame compute cost. V2 trades complexity for
per-pixel angular variation (sunset-side warmer than the opposite
horizon).

Total: ~5 working days for V1 (Phases 1-4); 3D aerial-perspective
LUT deferred to EN-005 V2.

## Risks

- **WebGPU 3D-texture support** for aerial-perspective. Confirmed
  supported in WebGPU spec; web target should work. If not, fall
  back to a 2D atlas.
- **Perf budget on web.** The 3D LUT is 32³×8 bytes = 256 KB; per-frame
  re-bake on sun move is fine, but a game that sweeps the sun every
  frame (real-time TOD) will pay ~1 ms on integrated GPUs. Document.
- **Banding** in the zenith on Rgba16Float. Standard fix is blue-noise
  dither in the sky shader; budgeted in Phase 4.
- **Existing panorama games regress.** Mitigated by gating on
  `setProceduralSky` — the old path is untouched.

## Acceptance

- Calling `setProceduralSky()` + `setSunDirection([0.3, 0.4, 0.7])`
  produces a noon sky with visible sun disk and correct directional
  light without any other setup.
- Sweeping the sun from zenith to horizon visibly warms the
  directional light, the sky, and the sun-shafts in lockstep.
- A game that doesn't call `setProceduralSky` renders identically to
  pre-RFC (panorama path unchanged).
- Headless screenshot tests for noon / sunset / pre-dawn pass on
  macOS-Metal.

## Decisions (2026-04-26)

1. **LUT sizes** — desktop keeps Hillaire defaults; web + mobile use
   the smaller tier shown in the LUT table. Tier picked at compile
   time via `cfg`.
2. **Cubemap re-bake resolution** — 64×6 across all platforms.
3. **API name** — `setProceduralSky(opts?)`.
