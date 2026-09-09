# Bloom renderer performance qualification

`bloom-render-perf` runs a fixed Ultra static scene through the production
headless renderer. Its primary P50/P95/P99 `cpu_render_submit_ms` measurement
covers the complete fixed renderer frame: `begin_frame`, renderer draw/light
submission, and `end_frame`, before any FPS cap. This includes upload work done
by renderer API setters instead of starting the clock after that work.
`cpu_prepare_ms` and `cpu_end_frame_ms` are also emitted as diagnostics. The
tool disables Bloom's unrelated default audio/physics/model-loading features
so comparison worktrees do not depend on optional submodules and both
revisions compile the same renderer-only workload.

Device creation uses the engine's production bounded negotiation and fallback
path. Every report embeds the complete adapter, renderer-tier/path, granted
feature/limit, selected device request, and fallback-cause snapshot under
`adapter`; a performance number without that capability evidence is not a
qualified result.

The optional `--trace-dir` mode enables wgpu's API trace only in this tool and
sums every traced buffer/texture upload between the final measured submits.
Never use trace-mode timings as performance evidence: the report marks them as
including trace I/O. Run an untraced command for timing and a separate short
traced command for upload volume.

When the tool is cherry-picked onto an older comparison revision, set
`BLOOM_RENDER_PERF_ENGINE_REVISION` to that engine commit so the JSON preserves
the actual code-under-test identity rather than the instrumentation commit.

```sh
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 180 --frames 300 \
  --out tools/quality/out/render-perf/1080p.json

cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 32 --frames 32 \
  --trace-dir tools/quality/out/render-perf/trace-1080p \
  --out tools/quality/out/render-perf/1080p-uploads.json
```

The workload defaults to `--quality-preset 4`. Use
`--quality-preset 0..4` to measure a complete tier, or add
`--render-scale 0.15..1.0` after the preset to compare renderer revisions at
an identical shading resolution. Reports record both the requested preset and
the effective scale, plus the renderer-path/resource snapshot taken after the
measured window.

## Visibility-buffer A/B workloads

The default `static-ultra` workload uses immediate geometry and therefore does
not qualify the retained GPU-driven visibility path. Two explicit retained
workloads exercise that path with the same 32x18 opaque grid:

- `visibility-low-overdraw` submits one layer (576 draws);
- `visibility-layered-overdraw` submits eight depth-separated layers (4,608
  draws), with the front layer covering the same screen area.

Run matched uncapped, timestamped processes because
`BLOOM_VISIBILITY_BUFFER` is selected during renderer initialization:

```sh
BLOOM_VISIBILITY_BUFFER=off cargo run --release \
  --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 180 --frames 300 \
  --quality-preset 0 --render-scale 1.0 \
  --workload visibility-low-overdraw --profile-passes \
  --out tools/quality/out/visibility-perf/low-off.json

BLOOM_VISIBILITY_BUFFER=shade cargo run --release \
  --manifest-path tools/render-perf/Cargo.toml -- \
  --width 1920 --height 1080 --warmup 180 --frames 300 \
  --quality-preset 0 --render-scale 1.0 \
  --workload visibility-low-overdraw --profile-passes \
  --out tools/quality/out/visibility-perf/low-shade.json
```

Repeat for `visibility-layered-overdraw` and use an alternating order (ABBA)
when making a performance claim. The report's `workload`, capability snapshot,
GPU-driven draw counts, visibility runtime state, `gpu_frame_*`,
`depth_prepass`, and `main_hdr_pass` fields are the qualification contract.
Shade mode folds ID rasterization into `depth_prepass` and visibility PBR into
`main_hdr_pass`, so compare those complete pass totals; there is deliberately
no artificial pass split merely to make the two stages easier to time.
