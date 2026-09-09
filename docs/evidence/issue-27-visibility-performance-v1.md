# Issue #27 visibility-buffer performance qualification v1

This checkpoint adds and runs two retained GPU-driven workloads against exact
revision `0cce864179cfe1e2d0cbf973784b2eb91df4cf58`. It measures the real
forward/off and visibility/shade paths instead of inferring performance from
render-target byte counts. No production renderer pass, shader, resource, or
activation default changed.

## Protocol

- Apple M1 Max integrated GPU, Metal, negotiated high-end tier;
- 1600x900 output and native `1.0` render scale;
- quality preset 0 to isolate geometry/depth/HDR composition from unrelated
  temporal and GI work;
- 180 warm-up frames and 240 measured uncapped frames per process;
- GPU timestamp queries enabled;
- process-isolated `BLOOM_VISIBILITY_BUFFER=off|shade` runs in ABBA order;
- two workload shapes:
  - low overdraw: one 32x18 opaque layer, 576 admitted draws;
  - layered overdraw: eight depth-separated 32x18 opaque layers, 4,608
    admitted draws over the same covered area.

Every shade sample reported visibility shading enabled, routed eligible and
compatibility indirect streams, every submitted draw admitted, zero
compatibility draws, and no fallback. Every off sample remained
forward-authoritative and allocated no visibility target or routed stream.

## Results

The table averages the two same-mode samples. `depth_prepass` contains the
inline ID raster in shade mode; `main_hdr_pass` contains the inline full-screen
visibility PBR evaluation. Those complete passes are intentionally measured
rather than split into artificial extra passes.

| Workload | Mode | GPU frame mean | GPU frame p95 | Depth mean | HDR mean |
|---|---:|---:|---:|---:|---:|
| 576 draws / one layer | off | 0.927790 ms | 0.955480 ms | 0.206709 ms | 0.281373 ms |
| 576 draws / one layer | shade | 1.024298 ms | 1.049417 ms | 0.207541 ms | 0.299204 ms |
| 4,608 draws / eight layers | off | 2.535183 ms | 2.868125 ms | 1.181747 ms | 0.755443 ms |
| 4,608 draws / eight layers | shade | 2.484490 ms | 2.663354 ms | 1.112699 ms | 0.806813 ms |

Visibility shading regressed the low-overdraw GPU mean by **10.40%** and p95
by **9.83%**. On the layered workload it improved GPU mean by only **2.00%**
and p95 by **7.14%**. The layered depth/ID stage was 5.84% faster, but the
visibility PBR portion of `main_hdr_pass` was 6.80% slower, leaving only the
small total-frame mean improvement.

## Decision

The opt-in path remains fail-closed. It does not meet either activation guard:

- low-overdraw mean must be no worse than 5% above forward; actual is +10.40%;
- the admitted layered stress mean must improve by at least 5%; actual is
  -2.00%.

This result is consistent with Bloom's alpha-aware depth prepass: hidden
fragments are already rejected before forward PBR, so a visibility buffer does
not recover the classic deferred-overdraw saving. The current full-screen
triangle reconstruction adds storage fetch and manual interpolation cost to
the winning pixels. Future optimization must reduce that measured
reconstruction/shading cost or target a workload with a demonstrated total
frame win; attachment-size arguments alone are not sufficient.

The permanent MRT/image oracle remains green independently. This performance
result rejects default activation; it does not reject the visibility path as a
correctness oracle or as the required shading route for virtual geometry.

## Reproduction

Use the commands in `tools/render-perf/README.md` for each workload and repeat
them in `off, shade, shade, off` order. The eight raw reports used here were
written under `/tmp/bloom-visibility-perf-0cce864/`; the stable aggregate is
recorded in `issue-27-visibility-performance-v1.json`.
