# Issue #131 candidate/visible scaling evidence v1

This checkpoint qualifies virtual-geometry culling and submission scaling at
revision `219344b3ef9d1d2bebcbc267eb90764c8e1a6892`. The permanent driver
uses one immutable 10,000,000-source-triangle archive for all points, changes
only the submitted placement set, records uncapped GPU timestamps, and
requires candidate/selected work to scale without disproportionate hierarchy
cost.

## Fixed archive and sweep

All three points use the same recipe-v2 artifact:

- 10,000,000 source triangles;
- 245,500 clusters and 8,496 pages;
- 582,052,704 bytes;
- artifact SHA-256
  `13ab0d150328b9f67ebc8abbe0d73891a10d72033bc3a18b4d9f6a8364cbd864`;
- index SHA-256
  `5c69a9a6cbf979d4c8483b3d8adc980cbe23d73862d5a79457051fe626d053c8`.

Reduced points select deterministic placements nearest the unchanged camera
target. They retain the complete archive, root table, GPU allocations, and
camera motion. Consequently the one-instance point still has 10M source
triangles available, but source-filtered root spans dispatch only work owned
by that placement.

| Instances | Candidate groups | Selected clusters | Resident pages | Selector GPU | Draw emission GPU | GPU frame | Wall frame |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 71 | 65 | 0.9561 ms | 0.0770 ms | 1.5373 ms | 4.0803 ms |
| 10 | 40 | 710 | 148 | 1.0586 ms | 0.0928 ms | 2.1805 ms | 5.9536 ms |
| 100 | 440 | 10,220 | 955 | 1.4661 ms | 0.1082 ms | 4.1692 ms | 6.0774 ms |

Across the sweep, instances grow 100x, candidate groups 110x, and selected
clusters 143.94x, while hierarchy-selection GPU mean grows only 1.53x. The
one-instance point's four visited groups, despite the unchanged 245,500-
cluster archive, directly guards against reverting to an archive-wide source
triangle/cluster scan. The full point remains below its established 3.0 ms
selector and 0.5 ms draw-emission limits.

## Fixed-budget runtime result

The full point warms streaming for 180 moving-camera frames and profiles 120
further moving-camera frames at 640x360 on Apple M1 Max Metal. It retains the
fixed 64 MiB physical pool, settles at 955 pages, completes 370 file requests
with zero failures, and reports zero fallback groups, missing-current pages,
selected/request overflow, invalid records, depth-limit fallbacks, and pending
streaming groups. GPU mean is 4.1692 ms, GPU p95 is 6.9526 ms, and wall mean is
6.0774 ms.

## Regression gate

`tools/quality/virtual_geometry_stress.py` now runs the 1/10/100 sweep for
Metal, Vulkan, or Direct3D 12 after one deterministic cook. It fails when:

- any point changes the source-triangle count or archive topology;
- a point submits no candidate or selected geometry;
- candidate/selected growth no longer tracks the requested instance growth;
- hierarchy-selection timing grows disproportionately to candidate groups;
- the requested backend is not the backend actually selected; or
- any established 10M timing, residency, streaming, overflow, or visual
  assertion fails.

The complete driver passed from a clean cook. Strict renderer
correctness/suspicious/performance Clippy and all 58 quality-tool tests passed.
The file-size ratchet remains red only for the same three unrelated
pre-existing files and this checkpoint adds no overage.

Discrete Vulkan and Direct3D 12 results remain pending until the repository's
self-hosted hardware runners return online.
