# Lumen-style real-time GI — multi-phase roadmap

UE5 Lumen-style global illumination for Bloom, with both a software (screen-space /
SDF) trace path and a hardware (ray-query) trace path. Scope is large (~1-2 months
minimum). This file is the table of contents; each phase is a self-contained ticket.

## Design target

Match UE5 Lumen's core structure, scaled down to Bloom:

- **Screen Space Radiance Cache (SSRC)** — one probe per 16×16 pixel block, 8×8
  octahedral atlas per probe, 32 rays/probe/frame, spatial filter + temporal
  accumulation (~4 frames), bilateral per-pixel reconstruction.
- **World Space Radiance Cache (WSRC)** — 3D clipmap probes at 32×32 octahedral
  resolution, sampled when SSRC rays travel > 2 m without hitting. Phase 3.
- **Two interchangeable ray backends** driving the same probe structure:
  - **SW**: Hi-Z trace (Phase 1) → Surface Cache + per-mesh SDF / global SDF
    clipmap (Phases 2-3).
  - **HW**: wgpu `EXPERIMENTAL_RAY_QUERY` with BLAS/TLAS (Phase 1) → shading
    upgraded by Surface Cache (Phase 2).

## Phase structure

Phase 1 ships SW + HW in parallel as one conceptual milestone, unblocked by a
mandatory wgpu version bump. The HW path is on the critical path, not deferred.

| Phase | Ticket | Scope | Parallelism |
|---|---|---|---|
| 0 | [007-prep](007-prep-wgpu-upgrade.md) | Bump wgpu from 24 to a release with Metal ray-query; sweep breaking changes across all 8 wgpu-using crates. | Serial prereq |
| 1a | [007a](007a-lumen-screen-probes-sw.md) | SSRC probe infrastructure + SW Hi-Z trace. Replaces current per-pixel SSGI. | Parallel with 007b |
| 1b | [007b](007b-lumen-screen-probes-hw.md) | BLAS/TLAS lifecycle + `rayQueryProceed` trace variant + hit-lighting-lite shading; runtime fallback to 007a on adapters without RT. | Parallel with 007a, merges after |
| 2 | [013](013-lumen-surface-cache.md) | Mesh Cards — 6-axis card capture at model load, per-frame card lighting pass, hits shade from card atlas. Upgrades HW quality to full Lumen. | After 007a+007b |
| 3 | [014](014-lumen-mesh-sdfs.md) | Per-mesh SDFs + global SDF clipmap + WSRC 32×32 clipmap probes. SW-only path reaches feature parity with HW. | Landed (V1-V15) — the SDF clipmap is now the default SW tier once baked |
| 5 | [016](016-lumen-importance-sampling.md) | Prev-frame radiance-guided ray direction sampling + hierarchical probe refinement in high-variance tiles. | After 013 |

Phase 4 from the original proposal was absorbed into Phase 1 (HW is no longer
deferred), so numbering skips directly from 013/014 to 016.

## Platform matrix

> **As-built correction (2026-07):** the "HW path" column below describes
> *capability*, not the default. The HW ray-query path is **opt-in**
> everywhere (`BLOOM_HW_GI=1` / `BLOOM_PT` / `--pt`); SW GI is the shipping
> default on every platform. See the as-built note under the table.

| Platform | SW path | HW path (opt-in) | Notes |
|---|---|---|---|
| macOS | yes (Phase 1a) | capable after 007-prep (Metal ray-query) | Unblocked by wgpu upgrade |
| iOS | yes | capable after 007-prep | Same |
| tvOS | yes | capable after 007-prep | Same |
| Windows | yes | capable (DXR) | Works post-upgrade; opt-in only |
| Linux | yes | adapter-gated (Vulkan ray-query) | Runtime feature check; SW fallback |
| Android | yes | adapter-gated | Most Android GPUs lack RT; SW expected in practice |
| Web | yes | never | No WebGPU RT spec; SW-only permanently |

**As-built (2026-07):** all phases landed (see the [README](README.md)
table). At runtime the probe trace selects a tier per frame:
`hw-ray-query` > `sdf-clipmap` > `hiz-screen` (`renderer/ssgi_pass.rs`).
Contrary to the matrix above, the HW path is now **opt-in**, not
auto-enabled on capable adapters: on Windows the ray-query device feature
is only requested when launched with `BLOOM_HW_GI=1`, `BLOOM_PT`, or
`--pt` (66dad5b) — the measured A/B showed +20 ms/frame on the 760M for a
tonal-only visual difference, so SW GI is the shipping default
(b34c1f3, 9d1523f).

## Per-phase acceptance protocol

Every ticket follows the standard perf-ticket protocol from [README.md](README.md):

1. Measure before/after with `--fps-only 300`.
2. Capture reference screenshot via `--capture 30 /tmp/<name>.png`; diff against
   baseline.
3. `./main --quality 0 --fps-only 60` regression guard must hit 60.
4. Commit code + PNGs together.
5. Update `README.md` ticket table + ms-budget line.

## HW shading quality caveat

007b ships **hit-lighting-lite** at BVH intersections: flat per-instance albedo ×
sun direct (shadow-map sampled) + analytic skylight. Textures at the hit point
require either Mesh Cards (Phase 2) or bindless vertex/texture fetch (out of
scope). Visual acceptance for 007b judges the *structural* win — off-screen
occlusion + off-screen bleed — not textured-hit quality. Phase 2 closes that gap.

## References

- Skorobogatova, "Real-Time Global Illumination in Unreal Engine 5" (Masaryk
  2022) — primary spec source.
- Epic, "Lumen Technical Details", SIGGRAPH 2022 Advances course.
- Majercik et al., "Dynamic Diffuse Global Illumination with Ray-Traced
  Irradiance Fields", JCGT 2019 — probe interpolation math.
- UE 5.4 source: `LumenScreenProbeGather.cpp`, `LumenSceneRendering.cpp`.
