# Bloom renderer qualification

This directory is the canonical visual-correctness and steady-state GPU/CPU
qualification workflow for Bloom. It turns a renderer change into reviewable
evidence: versioned scene inputs, fixed cameras and timesteps, native adapter
metadata, final and intermediate images, perceptual diffs, per-pass GPU
timestamps, and explicit pass/fail results.

It is a regression oracle, not a claim that the current images have reached
the project's quality target. In particular, the initial corpus deliberately
records visible material, temporal, and post-processing defects so later work
can prove that it improves them without breaking another scene.

## Commands

From the repository root:

```sh
# Fast local gate: PBR spheres, Damaged Helmet, and Sponza.
python3 tools/quality/run.py run quick

# Full nine-case hardware qualification.
python3 tools/quality/run.py run full \
  --machine-class apple-m1-max-metal

# Explore on an unqualified machine without changing the process exit code.
python3 tools/quality/run.py run full \
  --report-only \
  --out tools/quality/out/local-full
```

`run` is strict by default and exits non-zero for a missing asset/baseline,
capture error, missing intermediate, visual threshold failure, or applicable
hard performance budget. `--report-only` records those same failures in
`result.json`; it only makes the process exit zero for local investigation.
It never turns a failure into a recorded pass.

Useful focused commands:

```sh
python3 tools/quality/run.py check
python3 tools/quality/run.py run full --case bistro-exterior --report-only
python3 tools/quality/run.py faults
python3 -m unittest tools/quality/test_run.py -v
```

The output directory contains:

- `result.json`: the authoritative machine-readable result;
- `summary.md` and `summary.html`: compact human summaries;
- `cases/<id>/final.png`: the exact offscreen final output;
- `cases/<id>/intermediates/*.png`: named render-graph evidence;
- `metrics.json`, `heatmap.png`, and `comparison.png` when a baseline exists;
- complete prepare/build/capture/diff command records, including stdout,
  stderr, duration, exit status, and timeout state.

`tools/quality/out/` is ignored. CI uploads it as an artifact; do not commit
run directories.

## Determinism and measurement contract

Each case in `scenes.toml` versions its assets, camera, resolution, quality
tier, render scale, warm-up count, measured count, seed, timestep, required
features, thresholds, and performance budgets.

The scene executable uses `bloom/quality`'s `QualityRun`:

1. run a fixed-step warm-up with shader compilation excluded;
2. reset profiling and run the measured frames uncapped;
3. stop profiling before any readback;
4. request the final and named-intermediate capture;
5. serialize native telemetry and exit.

Headless qualification uses an exact-size offscreen target. It does not depend
on Retina/window scaling, desktop compositing, or vsync. The engine reports
`uncapped`, `present_mode`, warm-up exclusion, shader-compilation exclusion,
and GPU timestamp availability. Hard performance runners reject telemetry
that cannot prove these properties.

The measured window is intentionally sustained (240–300 frames after
120–180 warm-up frames). Shorter windows were rejected because scheduler
spikes dominated p95. Capture and PNG encoding happen after measurement.

The full corpus also includes two focused composition/silhouette cases:

- `weighted-transparency` exercises 96 intersecting imported BLEND layers;
- `masked-alpha-coverage` exercises 48 imported MASK cards across projected
  mip sizes, with deterministic object motion and cutout shadow casters.

The weighted fixture also accepts `--sorted` and `--refractive` for focused
backend validation of the two other temporal-reactive writers. These flags do
not alter the versioned default corpus. See
[`docs/temporal-reactive-coverage.md`](../../docs/temporal-reactive-coverage.md).

The `--refractive` route also exercises lazy transmitted directional shadows
when directional shadows are enabled. Native telemetry reports
`renderer_paths.transmitted_shadows` with the route's enable/active state,
`nearest-layer-rgb-depth` representation, fixed map resolution, exact
persistent bytes when allocated, and submitted caster count. Set
`BLOOM_TRANSMITTED_SHADOWS=0` for the exact visual/performance A/B control.
The ordinary no-transmission plan must have no transmitted-shadow graph
resources or pass. See
[`docs/transmitted-shadows.md`](../../docs/transmitted-shadows.md).

When SSGI and a retained transmission instance are both present, native
telemetry also reports `renderer_paths.transparent_gi`: enable/active state,
the bounded `one-layer-colored-continuation` representation, exact additional
persistent bytes (zero), and retained instance count. Set
`BLOOM_TRANSPARENT_GI=0` for the opaque-GI A/B control. Opaque scenes must not
create or select the lazy hardware/SDF/WSRC pipeline specializations. See
[`docs/transparent-gi.md`](../../docs/transparent-gi.md).

`examples/quality-transparency/main.ts` accepts `--transparent-gi` as an
unversioned focused stress route. It replaces the 96 immediate draws with 96
moving retained physical-transmission nodes and disables their independent
directional-shadow contribution. This makes an environment-variable on/off
run measure the GI specialization while keeping camera refraction, transforms,
Mesh-Cards, TLAS rebuilds, and scene composition identical. The flag does not
alter the versioned default corpus.

The same fixture accepts `--reflection-hierarchy` as an unversioned focused
glass-reflection oracle. It creates an explicit horizontal planar probe and a
smooth imported-transmission floor reflecting a rotating Damaged Helmet.
Native telemetry reports `renderer_paths.refractive_reflections`, including
the planar/screen-space/environment source order, fixed march bounds, the
lazy 160-byte uniform, and zero additional graph passes/images. Set
`BLOOM_REFRACTIVE_REFLECTIONS=0` for the exact environment-only A/B control.
The flag does not alter the versioned default corpus. See
[`docs/refractive-reflections.md`](../../docs/refractive-reflections.md).

`renderer_paths.physical_texture_uv` reports supported UV sets, lazy
TEXCOORD_1-pipeline initialization, the unchanged 96-byte ordinary vertex
stride, the 8-byte UV1 sidecar stride, and zero graph/image cost.

`renderer_paths.transparency` also reports the
`global-depth-source-stable-id` conventional sorted-interleaving contract,
the number of lazy attachment-compatible custom pipelines initialized so far,
and the invariant zero additional draws/graph passes. Weighted OIT remains an
imported aggregate resolved before custom commands.
`BLOOM_SORTED_INTERLEAVING=0` restores the prior list boundary for exact A/B.

`examples/quality-transparency/main.ts` accepts `--sorted-interleaving` as an
unversioned focused ordering oracle. It forces conventional sorted composition
and pairs the 96 imported BLEND layers with 96 custom-material layers at
alternating depths while TAA is active. This makes the global order and lazy
reactive-compatible custom pipeline observable; the environment kill switch
above supplies the identical-scene legacy control. The flag does not alter the
versioned default corpus.

### Native evidence

The native renderer reports the adapter name, vendor/device IDs, device type,
driver fields when exposed by wgpu, backend, capability tier, semantic feature
set, actual SSGI trace backend, and path-tracing availability. Empty driver
strings are valid on Metal because the backend does not expose them; they are
recorded rather than invented.

The one debug-capture API snapshots existing render-graph products:

- `hdr-scene`: RGBA16F scene output, converted for review with the same
  ACES-style display curve used by the capture helper;
- `ssgi`: RGBA16F resolved indirect diffuse output, accompanied by raw HDR
  finite/luminance metrics;
- `scene-depth`: Depth32F normalized to the finite range of that capture for
  diagnostic visibility (not a metric-preserving linear-depth encoding);
- `shadow-cascade-0`, `shadow-cascade-1`, `shadow-cascade-2`: Depth32F shadow
  maps normalized per cascade.

The API is dormant during ordinary rendering and throughout the measured
window. It reuses textures already marked `COPY_SRC` and records copies in a
keyed terminal render-graph pass during the post-measurement screenshot
submission. It creates no normal-frame pass, allocation, bind group, or
readback.

To retain the exact normal and capture plans beside a qualification result,
set `BLOOM_GRAPH_DUMP_DIR` to an absolute directory. Each distinct plan emits
deterministic JSON and DOT once. The schema, cache counters, lifetime fields,
and aliasing constraints are described in
[`docs/compiled-render-graph.md`](../../docs/compiled-render-graph.md).

## Reproducibility

Run the same suite twice without changing source, configuration, machine
power mode, or background GPU load, then compare the bundles:

```sh
python3 tools/quality/run.py run quick \
  --report-only --out tools/quality/out/repro-a
python3 tools/quality/run.py run quick \
  --report-only --out tools/quality/out/repro-b
python3 tools/quality/run.py repro-check \
  --first tools/quality/out/repro-a/result.json \
  --second tools/quality/out/repro-b/result.json
```

`repro-check` requires identical stable metadata and artifact sets. It hashes
every final/intermediate PNG and applies tighter-than-regression image metrics
when hardware ray-query sampling is not byte-identical. It compares CPU/GPU
mean and p95 against the versioned `[reproducibility]` bounds.

The current same-machine contract is:

- final/intermediate SSIM at least `0.999`, luminance RMSE at most `0.002`,
  mean OKLab and edge deltas at most `0.001`;
- CPU/GPU mean: 15% relative noise, with small absolute allowances of
  0.35 ms CPU and 1.0 ms GPU;
- CPU p95: 25% or 1.0 ms;
- GPU p95: 50% or 12 ms because OS/GPU scheduling can move a small number of
  frames across the 95th-percentile boundary.

The wider p95 reproducibility envelope does not weaken hard budgets: a
qualified machine still compares the observed absolute p95 to the case budget.
If a reproducibility run fails, remove background load and retry; do not raise
bounds without attaching multiple-run evidence.

## Machine classes and fallbacks

Hard budgets apply only when `--machine-class` selects the class declared by
that case. The runner verifies the native adapter/backend against the selected
class, so a label or environment variable cannot impersonate a qualified GPU.

Defined classes:

- `apple-m1-max-metal`: Metal high-end RT/bindless baseline;
- `nvidia-rtx4080-vulkan`: Vulkan discrete high-end baseline, including VRAM;
- `apple-m1-metal-constrained`: constrained preset/render-scale gate.

Runs without a machine class still record timings and visual failures, but
performance is report-only. VRAM must come from the hardware runner when wgpu
cannot report it:

```sh
export BLOOM_QUALITY_VRAM_PEAK_MB=2450
# or case-specific, with punctuation converted to underscores:
export BLOOM_QUALITY_VRAM_PEAK_MB_BISTRO_EXTERIOR=6120
```

Required features are evaluated again from the native runtime probe. Sponza
and Bistro record their declared software GI fallback when ray query is not
available; `feature_decision` says `native`, `fallback`, or `unsupported`, and
telemetry records the path actually used.

## Baseline governance

A normal run never creates or overwrites a baseline. Create a review bundle:

```sh
python3 tools/quality/run.py baseline-review \
  --result tools/quality/out/local-full/result.json \
  --out tools/quality/out/baseline-review \
  --reason "Explain the renderer change and expected visual effect"
```

The bundle contains the proposed image, current image when present, available
diff/heatmap/metrics, all named intermediates, timing evidence, source commit,
manifest hash, and reason. For an initial baseline it records
`baseline_state: absent`.

After a human reviews the final and intermediate images, install explicitly:

```sh
python3 tools/quality/run.py baseline-install \
  --review tools/quality/out/baseline-review/review.json \
  --approved-by "Reviewer Name" \
  --receipt tools/quality/out/baseline-review/installation.json
```

Installation is restricted to `tools/quality/baselines/`. It refuses a stale
review if the baseline changed after the bundle was made, and it refuses to
replace a newly-created target when the review expected no baseline. Do not
use an agent/service identity for `--approved-by`; this is the independent
visual-review boundary.

A baseline PR must provide:

- why the pixels should change;
- before/after/comparison/heatmap (or explicit initial-baseline status);
- SSIM, RMSE, OKLab, and edge deltas;
- CPU/GPU p95 deltas on the applicable machine class;
- the human reviewer and installation receipt.

Backend-specific baselines require a documented, reproducible raster or
floating-point difference that cannot be normalized. Portable baselines are
the default.

## Seeded negative controls

`python3 tools/quality/run.py faults` asks `bloom-diff` to create five
deterministic corruptions from approved baselines and succeeds only when every
one is rejected:

- BRDF energy;
- shadow placement;
- GI leakage;
- motion history;
- texture orientation.

This proves that passing thresholds still detect the failure classes the
corpus is intended to guard. Fault images, metrics, and commands are retained
under `tools/quality/out/faults`.

During initial bring-up, before any baseline has received human approval, the
detector itself can be demonstrated against a complete result bundle:

```sh
python3 tools/quality/run.py faults \
  --source-result tools/quality/out/local-full/result.json \
  --out tools/quality/out/bootstrap-faults
```

That result is explicitly labelled `unapproved-result-bundle`; it proves fault
detection but does not substitute for approved-baseline CI. Scheduled hardware
CI intentionally omits `--source-result`.

## Corpus notes

- PBR spheres: high and constrained BRDF/material-lobe contracts.
- Damaged Helmet: canonical glTF textures, UVs, tangent normal map,
  metallic-roughness, occlusion, and emissive behavior.
- Sponza: interior GI, cascaded shadows, alpha leakage, and temporal output.
- Bistro: exterior sun/sky and representative large-scene materials.
- Skinned alpha motion: looping Fox deformation in front of textured,
  alpha-tested Sponza foliage.
- Draw/light stress: 10,240 meshes plus many lights.
- Weighted transparency: 96 imported BLEND layers in 12 animated,
  intersecting cells; records OIT accumulation/resolve cost and stability.

The checked-out Bistro source has 2,909 mesh-node instances and 551 unique
meshes. Bloom's current `ModelData` ABI duplicates vertex arrays per instance;
loading the literal scene consumed about 19 GB RSS and exceeded the 256 MB
per-buffer limit when building ray-query geometry. Until GPU instancing lands,
`prepare_bistro.py` deterministically selects 96 largest camera-visible unique
meshes, preserves their authored world transforms/materials/textures, and
records source/derived hashes. It reads `bistro.bin` from the pinned Git
revision into the case output and never edits the working-tree asset. The
10k-mesh stress case separately covers draw pressure. Do not describe the
current Bistro case as full 2,909-instance streaming coverage.

## CI and migration

`.github/workflows/quality.yml` runs contract tests on hosted CI and runs the
full suite plus negative controls on labelled hardware runners. Evidence is
uploaded with `if: always()`, and `summary.md` is appended to the job summary.

This workflow supersedes `tools/validate.sh` as the regression gate.
`validate.sh` remains a legacy, macOS-only four-camera Helmet exploration tool;
its cached reference, `sips` resize, embedded cameras, vsync/warm-up behavior,
and report-only output are not a qualification contract.

Existing tools are reused:

- `bloom-reference` remains the deterministic CPU/path-traced reference
  generator and PT oracle;
- `bloom-diff` is the authoritative visual metric/fault engine;
- `tools/cycles_reference` remains useful for offline artistic/reference
  studies;
- their approved outputs enter qualification only through an explicit
  baseline review.

When adding a case, update `scenes.toml`, pin every asset with hash/revision
plus source/license, implement the same `QualityRun` CLI contract, capture at
least final/HDR/depth evidence, add it to the appropriate workflow list, and
add or update a negative control when it represents a new failure class.
