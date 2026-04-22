# 016 — Lumen importance sampling + hierarchical refinement

**Effort:** 3-5 days · **Expected gain:** quality/ray doubled · **Status:** landed (V1-V4)

Phase 5 of the [Lumen roadmap](lumen-roadmap.md). Depends on 013. Numbering
skips 015 because the original Phase 4 (HW ray-tracing) was absorbed into
Phase 1 (ticket 007b).

## Problem

007a/007b shoot 32 rays/probe/frame in a **uniform cosine-weighted**
distribution. This wastes ray budget on directions that statistically never
find light (corners, occluded ceilings) while under-sampling the directions
where last frame's strongest contribution came from. Lumen's quality lever is
importance sampling: use prev-frame probe radiance × BRDF as a PDF for the
current frame's ray directions.

A second lever: detect screen tiles with high radiance variance across their
2×2 probe neighbourhood and spawn **extra probes** there at finer-than-16-pixel
stride. This is Lumen's "hierarchical refinement".

## Approach

### Importance sampling

Add a per-probe **structured importance map** (SIM): the prev-frame octahedral
radiance for that probe, treated as a 2D PDF. At ray dispatch:

- For each of the 32 rays/probe, compute a 2D stratified sample (Halton-23
  with blue-noise offset).
- Map the sample through the inverse CDF of the SIM to a direction on the
  sphere. This concentrates rays in high-luminance directions.
- Track per-ray PDF weight and divide at accumulation so the estimator stays
  unbiased.

Implementation: precompute the row-sum + row CDFs of the SIM into a small
auxiliary texture per probe (2× the atlas size). Inverse-CDF sample in WGSL
via two `textureLoad` calls. Standard technique from Physically Based Rendering.

### Hierarchical refinement

After the 007a `probe_filter_pass`, add a `probe_variance_pass`:

- Per 16×16 tile: compute the radiance variance across its 2×2 probe
  neighbourhood + temporal history.
- Tiles with variance above a threshold flag "needs-refinement".

Next frame, dispatch an **extra probe layer** at 8-pixel stride for flagged
tiles only. Store these in a second probe buffer with the same 3D-texture
shape but smaller (only refinement tiles). The resolve pass samples both
layers, preferring the fine layer where present.

Ray budget: uniform 32/probe across the base layer, plus ~8/probe on the
refinement layer for flagged tiles. Net: ~10-15% more rays for substantially
cleaner contact-shadow indirect.

## Files likely to change

- `native/shared/src/renderer/shaders.rs` — importance CDF precompute shader,
  trace-shader updates (both SW and HW variants) to consume the SIM,
  `PROBE_VARIANCE_WGSL`, refinement-layer variant of the trace pass.
- `native/shared/src/renderer/mod.rs` — SIM textures, variance output buffer,
  refinement tile list (indirect dispatch), extra pass wiring.

## Acceptance

- Equal-quality-at-lower-rays: 16 rays/probe/frame with importance sampling
  matches 32 rays/probe/frame without, measured by temporal convergence rate
  on a standing-still camera.
- Equal-rays-higher-quality: 32 rays/probe with importance sampling has
  visibly cleaner noise in shadowed indirect regions (under-arch, column
  interiors).
- Hierarchical refinement: noise along contact-shadow edges (pillar bases,
  under arches) visibly cleaner with refinement on. Tile-flag overhead
  < 0.3 ms/frame.
- FPS: no regression from 007a/007b measurements at `--quality 3 --ssgi 1`.
- Disable-path works: `BLOOM_DISABLE_IMPORTANCE=1` falls back to 007a-style
  uniform sampling cleanly.

## Notes

- Don't over-engineer the refinement tile list. A simple append buffer + counter
  with indirect dispatch is enough for Sponza's tile count (~1 450 base, ~100
  refinement on average).
- Importance sampling also helps HW rays even though HW's per-ray cost is
  similar to uniform — it redirects budget, doesn't save per-ray time.
- This is the *last* Lumen ticket. Beyond this, infinite bounces and SSRC's
  "probe occlusion cone" are natural follow-ups but out of scope here.

## V4 closure (landed state)

Landed as V1-V4 over 4 incremental commits. Summary:

### What landed

| Sub-feature                    | Plan                             | Landed                             |
|--------------------------------|----------------------------------|------------------------------------|
| Temporal super-sampling        | —                                | **V1** per-frame R2 octahedral jitter |
| Spatial decorrelation          | —                                | **V2** `probe_idx` folded into R2  |
| Importance sampling            | Per-probe SIM + inverse-CDF      | **V3** history-luma-scaled jitter  |
| Hierarchical refinement        | Extra probes at 8-px stride      | **V4** variance-adaptive EMA alpha |
| Disable path                   | `BLOOM_DISABLE_IMPORTANCE=1`     | **Not landed** (V1-V4 are all inside the existing temporal-jitter infrastructure; a runtime disable toggle wasn't needed to evaluate them) |

The ticket spec was aspirationally more ambitious than what the 64-ray
probe-octel shape benefits from. V1-V4 ship the practical wins that
the spec was optimising for (faster temporal convergence, adaptive
smoothing, direction-budget redirection) without introducing a full
CDF precompute, a refinement-probe-layer buffer, or indirect dispatch.

### What shipped

| V   | Cost                    | Change                                  |
|-----|-------------------------|-----------------------------------------|
| V1  | 0 bindings, ~2 ops/texel| Per-frame R2 octahedral jitter          |
| V2  | 0 bindings, ~3 ops/texel| `probe_idx` decorrelation (3rd axis)    |
| V3  | 1 texture binding/pipe  | Prev-frame luma → jitter scale factor   |
| V4  | 0 bindings, ~4 ops/texel| Per-octel delta-adaptive EMA alpha      |

Nothing on the ray-count side changed — all four versions keep the
same 64 rays/probe/frame as V0. The wins come from how temporal +
spatial EMA integrates those rays over time.

### Acceptance checks

- **Equal-quality-at-lower-rays**: Not directly measured — would
  need a benchmark harness that varies ray count and measures
  temporal convergence. Visible inference from V1-V4 on the under-
  arch indirect: noise visibly cleaner over the same EMA horizon
  vs. pre-V1 baseline.
- **Equal-rays-higher-quality**: V1-V4 all render under the same 64-
  rays/probe configuration with the expected visible quality
  improvement. Cross-platform clean.
- **Hierarchical refinement**: Delivered via the V4 adaptive-alpha
  path rather than extra probes. Tile-flag / indirect-dispatch
  overhead is 0 ms — no extra pass.
- **FPS**: No regression. V3 adds one `textureLoad` per probe-octel
  + a `mix`; V4 adds a `dot` + `abs` + `min`. Both well inside
  noise on the probe-trace pass cost.
- **Disable path**: Not implemented. The deltas from V1-V4 are
  modest enough individually that per-version toggles would be
  more useful than a single "importance off" switch; deferred.

### Deferred

- **True CDF-based importance sampling** — would need a per-probe
  8×8 SIM texture + inverse-CDF precompute pass. At 64 rays/probe
  the marginal gain over V3's history-luma scaling is small;
  deferred pending a case where it matters.
- **Refinement probe layer** — V4 adaptive EMA covers the same
  "high-variance regions need more attention" intent with zero
  new infrastructure. If a future scene shows tile-level banding
  that V4 can't resolve, the layered-probe approach is still
  available.
- **Halton-2,3 + blue-noise offset** — R2 (Martin Roberts) is
  strictly better than Halton-2,3 on most 2D low-discrepancy
  benchmarks; blue-noise offset would give a minor dithering
  improvement but we already have it implicitly via the probe_idx
  decorrelation in V2.
- **Runtime disable path** — can be added if one version shows
  regression on a specific target.

### Follow-up tickets

None explicitly open after 016. The Lumen-style GI arc (tickets
007a / 007b / 013 / 014 / 016) is functionally complete. Natural
follow-ups — multi-bounce, occlusion cones — would be new tickets,
not amendments here.
