# Bloom renderer performance qualification

`bloom-render-perf` runs a fixed Ultra static scene through the production
headless renderer. It measures only `EngineState::end_frame`, after draw
submission and before any FPS cap, and reports P50/P95/P99 CPU render-submit
time. The tool disables Bloom's unrelated default audio/physics/model-loading
features so comparison worktrees do not depend on optional submodules and both
revisions compile the same renderer-only workload.

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
