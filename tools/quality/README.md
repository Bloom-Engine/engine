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

To isolate an intermittent detailed-Bistro temporal defect by renderer owner,
capture the same admitted scene, camera excursion, return pose, and 32-frame
stationary burst with TAA, SSGI, SSR, and Hi-Z occlusion removed one at a time:

```sh
python3 tools/quality/bistro_temporal_matrix.py \
  --scene /absolute/path/to/BistroReference.gltf \
  --output /tmp/bloom-bistro-temporal-matrix \
  --max-largest-component-pixels 32
```

The command writes each variant's images and `metrics.json`, plus a root
`matrix.json` comparing temporal pixel range, adjacent-frame change, and the
largest coherent changing region. It is diagnostic evidence rather than a
quality threshold: the control reductions identify which subsystem should be
changed before a permanent regression gate is approved. Use `--analyze-only`
to recompute metrics without recapturing, or `--resume` after an interrupted
matrix to retain complete variants. The underlying opt-in golden also accepts
`BLOOM_BISTRO_PROBE_DUMP_OCCLUSION=0` for the exact visibility control and a
comma-separated `BLOOM_BISTRO_PROBE_DUMP_SEQUENCE_DIAGNOSTICS` list when TAA
policy images are needed at particular sequence frames.

The `32`-pixel coherent-component limit is the qualified 512×288 gate for
issue #151: the prior full path measures 55 pixels and the corrected path 25.
Keep the size and 32-frame sequence unchanged when applying that limit.

To qualify fractional TAA/TSR against a matched native-resolution camera
motion, capture the same detailed-Bistro pose at the candidate scale, its
comparison revision, and scale 1.0. The opt-in dump accepts
`BLOOM_BISTRO_PROBE_DUMP_RENDER_SCALE`, a linear `dx,dz,dyaw` in
`BLOOM_BISTRO_PROBE_DUMP_SEQUENCE_MOTION`, and
`BLOOM_BISTRO_PROBE_DUMP_DIAGNOSTICS=0` when only final frames are required.
Compare the resulting numbered sequences with:

```sh
python3 tools/quality/tsr_motion_compare.py \
  --baseline /tmp/bloom-tsr-baseline \
  --candidate /tmp/bloom-tsr-candidate \
  --native /tmp/bloom-tsr-native \
  --expected-frames 32 \
  --output /tmp/bloom-tsr-comparison.json
```

The command fails closed unless the candidate is no worse than its baseline
in both normalized RGB error against native frames and error against native
adjacent-frame motion derivatives. It also records exact dimensions and
endpoint hashes so differently posed or incomplete sequences cannot silently
be compared.

For the full-resolution Sponza motion gate, the example owns a deterministic
frame-indexed camera crawl and the runner captures native 1.0, fractional
0.75, and an independent fractional repeat:

```sh
python3 tools/quality/sponza_tsr_native_match.py \
  --output /tmp/bloom-sponza-tsr-native-match \
  --max-native-frame-rmse 0.0133 \
  --max-native-motion-derivative-rmse 0.0100
```

The runner forces a fixed timestep and pixel-exact headless output, rejects an
incomplete or incorrectly sized sequence, proves that the native negative
control actually moves, checks repeat captures against the manifest's
hardware-ray reproducibility bounds, and records both streamed RGB reference
errors and per-frame `bloom-diff` metrics. Use `--analyze-only --skip-build`
to re-evaluate an existing output directory without recapturing it. Captures
are diagnostic evidence; the command never installs or approves a baseline.

For the official Khronos alpha, transmission, volume, and ordering controls:

```sh
python3 tools/quality/khronos_materials.py \
  --out tools/quality/out/khronos-materials
```

That opt-in command downloads four `.glb` files from one pinned
`glTF-Sample-Assets` revision into the ignored output directory and verifies
their SHA-256 hashes. It builds `examples/renderer-test`, captures each case
twice, rejects supported-field importer/validation diagnostics, rejects
flat/black output, and records exact same-machine repeatability. Its
`summary.md` links the candidate images for review. It deliberately does not
install or approve a baseline; semantic reference-image approval remains a
human action.

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

`renderer_paths.steady_state_uploads.lighting` reports the last frame's
lighting-buffer `write_count`, actual `byte_count`, and `full_buffer_bytes`.
Lighting setters update one CPU snapshot; the renderer compares that snapshot
once before submission and emits at most three aligned dirty ranges (fixed and
directional fields, point lights, and view/shadow/frame data). This makes the
former repeated full-buffer upload observable without adding a readback or GPU
pass.

`renderer_paths.steady_state_resources.bind_group_creations` reports the last
frame's total bind-group creations and a fixed, named count for every recurring
core-frame site. The counter storage is a twelve-element integer array: it
performs no allocation and lets qualification distinguish true steady-state
churn from initialization, resize, or resource-generation rebuilds.
Final-composite bindings are cached across the exact Cartesian product of
eight possible source views and two exposure-history slots. Resize invalidates
all sixteen entries before replacing any referenced render-target view, so a
warmed stable path reports `final_composite: 0` without stale-view reuse.
Scene-compose bindings likewise use distinct slots for the cleared SSR
fallback and both SSR history views. A warmed stable path therefore also
reports `scene_compose: 0`; SSR toggles and path-tracing ownership select a
different complete binding instead of mutating or incompletely keying one.
SSR temporal bindings are also cached separately for the two alternating
previous-history inputs. The optional diagnostics pass consumes the same
cached binding, and resize invalidates both entries before history, raw SSR,
or velocity views are replaced.
Ordinary TAA uses the same two-slot history-keyed cache and reports `taa: 0`
after warmup. Reactive TAA remains a separately named counter and uses two
history slots keyed by both compiled plan ID and transient-pool rebuild epoch,
because its coverage view belongs to that compiled transient generation. It
also reports `taa_reactive: 0` after warmup and after a resize rebuild settles.
The non-TAA half-resolution upscale binding is a single persistent slot,
invalidated before resize replaces its composed input. A dedicated real-GPU
half-resolution test proves scene pixels are produced and `upscale: 0` is
restored after warmup.
DoF, motion blur, SSS, and CAS use lazy bind-group arrays indexed by the exact
upstream color target selected by the post-FX chain (including both TAA
history slots). Resize clears every array before replacing those views. A
forced full-chain GPU test renders geometry and hard-gates all four named
creation counters to zero after warmup.
Auto exposure uses sixteen lazy entries for the same eight composite-source
identities crossed with both previous-exposure slots. The forced full-chain
test enables exposure adaptation and also hard-gates `auto_exposure: 0` after
both ping-pong bindings are warm.
Each user post-pass owns two lazy bindings for the LDR A/B input parity and
drops them before resize replaces color/depth views. A two-pass copy stack
proves geometry preservation and `custom_post_pass: 0` both before and after a
resize cycle. With every named core site covered, official post-warmup quality
artifacts now fail the contract unless total bind-group creation is zero.

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
recorded rather than invented. Runtime evidence also reports
`ray_scene_preparation` as `disabled`, `ssgi`, `pt`, or `ssgi+pt`; this makes
the shared acceleration/card prefix observable without conflating it with
SSGI-only baking.

Every accepted native telemetry artifact must also contain the complete
`adapter.renderer_capabilities` and `adapter.device_negotiation` snapshots.
The quality runner validates tier identity, granted features/limits, selected
system paths, active platform profile, chosen request, fallback cause, and
requested device limits before comparing images or timings. Missing or
inconsistent capability evidence fails the case rather than producing an
unqualified performance result. Each run also writes the same evidence as the
named top-level `capabilities.json` artifact referenced by `result.json`; the
hardware workflows upload the entire result directory on both success and
failure.

The one debug-capture API snapshots existing render-graph products:

- `hdr-scene`: RGBA16F scene output, converted for review with the same
  ACES-style display curve used by the capture helper;
- `ssgi`: RGBA16F resolved indirect diffuse output, accompanied by raw HDR
  finite/luminance metrics;
- `scene-depth`: Depth32F normalized to the finite range of that capture for
  diagnostic visibility (not a metric-preserving linear-depth encoding);
- `shadow-cascade-0`, `shadow-cascade-1`, `shadow-cascade-2`: Depth32F shadow
  maps normalized per cascade.

Active temporal systems add capture-only evidence beside those physical graph
products. Realtime PT emits trace-resolution rejection reason, motion,
reprojected UV, and variance/history confidence in one temporary compute pass.
The normal renderer creates none of those resources; runtime telemetry reports
their exact temporary byte/pass contract and release state.

The manifest makes the complete TAA, SSR, and SSGI capture set mandatory for
every High-preset corpus case. A missing diagnostic therefore fails
qualification instead of silently producing an incomplete evidence bundle.
Lower presets require only the graph products that remain valid for their
disabled feature set. Realtime PT uses the same named-artifact contract when a
PT corpus case is enabled.

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

The virtual-geometry portability gate is separate from the ordinary scene
manifest because it deterministically generates and cooks a 10M-source-
triangle asset before rendering it. Run it with an explicit backend so the
test can assert the selected adapter path rather than merely accepting wgpu's
primary choice:

```sh
python3 tools/quality/virtual_geometry_stress.py \
  --platform macos --backend metal \
  --work /tmp/bloom-vg-stress-work \
  --out tools/quality/out/virtual-geometry-metal
```

The hardware workflow runs the same driver as `macos/metal` and
`linux/vulkan`, uploads the final frame and complete timing/residency/I/O
telemetry, and publishes a compact job summary. Each run also reuses the exact
10M archive for a 1/10/100-instance uncapped scaling sweep. The sweep requires
candidate groups and selected clusters to grow with the submitted instance
set while hierarchy-selection GPU time remains bounded by that candidate
growth; unrelated clusters elsewhere in the archive therefore cannot hide a
source-triangle scan. A future Windows hardware runner uses `--platform
windows --backend dx12`; the runtime gate already rejects a result if the
requested backend was not actually selected.

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

For interactive inspection, `scripts/run-bistro-rich.sh` prepares and launches
a separate non-governed profile containing all 551 unique source meshes at
their first authored transforms. Generated files stay under
`examples/bistro/.generated/`; the 96-mesh qualification corpus and its
baselines remain unchanged. This is substantially richer than the bounded
qualification view, but it is not a substitute for native instancing of every
authored node.

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
