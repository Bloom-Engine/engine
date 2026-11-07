# Issue #139 steady-state performance evidence

This qualification compares the last engine revision before the #139
steady-state optimization series (`0bc6fabc`) with the instrumented current
revision (`93dfd942`). The benchmark tool commits were cherry-picked unchanged
onto the detached before worktree; `BLOOM_RENDER_PERF_ENGINE_REVISION` preserves
the actual engine revision in each report.

## Controlled workload

- Apple M1 Max / Metal
- release build
- production headless renderer, FPS cap disabled
- Ultra preset 4
- static plane and cube with 40 colored point lights
- 300 warm-up frames, then 900 measured frames
- three runs per revision and resolution
- P50/P95/P99 below are medians of the three per-run percentiles
- primary `cpu_render_submit_ms` covers `begin_frame`, fixed renderer
  draw/light submission, and `end_frame`
- timing runs have wgpu API tracing disabled

## CPU render-submit result

| Resolution | Metric | Before (ms) | After (ms) | Change |
|---|---:|---:|---:|---:|
| 1920×1080 | P50 | 2.654 | 2.465 | −7.1% |
| 1920×1080 | P95 | 4.069 | 3.352 | −17.6% |
| 1920×1080 | P99 | 12.179 | 3.507 | −71.2% |
| 3840×2160 | P50 | 6.301 | 6.100 | −3.2% |
| 3840×2160 | P95 | 6.558 | 6.273 | −4.3% |
| 3840×2160 | P99 | 7.688 | 6.387 | −16.9% |

Renderer prepare mean fell from 0.261 to 0.047 ms at 1080p (−82.1%) and
from 0.217 to 0.061 ms at 4K (−71.7%). The separately reported 4K
`end_frame` mean was effectively flat (5.834 vs 5.839 ms, +0.005 ms /
+0.08%) because the headless three-frame queue moves GPU back-pressure into
that phase. The complete render-submit measurement, which is the acceptance
metric, improved at every percentile and resolution.

## Total upload result

A separate 32-warm-up / 32-measured-frame run enabled wgpu API tracing only in
the qualification binary. Every traced buffer and texture payload between the
final 32 submits was summed. Trace-mode timings are explicitly marked invalid
and were not used above.

| Resolution | Before | After | Change |
|---|---:|---:|---:|
| 1920×1080 | 422,120 B/frame | 23,264 B/frame | −94.5% |
| 3840×2160 | 422,120 B/frame | 23,264 B/frame | −94.5% |

All 32 measured frames at each revision/resolution had the exact reported
value; steady texture-upload bytes were zero. The optimized frame transfers
398,856 fewer bytes, an 18.14× reduction.

## Commands

Timing:

```sh
BLOOM_RENDER_PERF_ENGINE_REVISION=<revision> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width <1920-or-3840> --height <1080-or-2160> \
  --warmup 300 --frames 900 --out <report.json>
```

Upload volume:

```sh
BLOOM_RENDER_PERF_ENGINE_REVISION=<revision> \
cargo run --release --manifest-path tools/render-perf/Cargo.toml -- \
  --width <1920-or-3840> --height <1080-or-2160> \
  --warmup 32 --frames 32 --trace-dir <trace-directory> \
  --out <upload-report.json>
```

The raw values and aggregate calculation are preserved in
[`issue-139-steady-state-performance.json`](issue-139-steady-state-performance.json).
