# Issue #131 measured source error (geometry recipe v4) evidence

This checkpoint changes what a cooked hierarchy stores as a cluster group's
error, and stops a hierarchy at a stalled level. It changes cooked hierarchy
artifacts, so the geometry recipe version moves from 3 to 4. Leaf-only cooks
(no `--hierarchy-levels`) are unaffected.

## What was wrong

Recipes v2 and v3 stored `max(simplifier result, per-level vertex deviation)`
added to the children's error. The simplifier result includes the attribute
terms at the weights that order collapses (normals 0.5, UV 10, colour 1), and
the sum grows with every level.

| Asset | Size | Coarsest level | Recorded error, v3 | Measured against the source, v4 |
|---|---|---|---:|---:|
| `DamagedHelmet.glb` | 1.9 x 2.0 x 1.8 | level 3, 4,510 of 15,452 triangles | 2.468 | 0.339 |
| A game's house model (80,000 triangles, one 4K atlas, many UV islands) | 11.6 x 7.0 x 11.7 m | level 7, 16,779 triangles | 6.300 m | 0.297 m |

The helmet's recorded error was larger than the helmet. For the house, the
first level (40,100 triangles) was recorded at 0.03 to 1.0 m; measured, it is
0.03 to 0.12 m. With a one-pixel target at 3200x1800 and a 50 degree field of
view a 1.0 m error is first allowed about 1.9 km away, so every view of that
model drew its leaves: a hierarchy with no effect on the picture or the cost.

## What v4 stores

For each parent group: the largest distance from any level 0 vertex the group
replaces to the group's simplified triangle surface, times 1.0001 against
float rounding, and never less than its children's error. The cooker carries
each group's source positions up the hierarchy and queries the existing BVH
(`geometric_error.rs`). The hierarchy oracle test, which already measured
every level 0 descendant against every parent surface, now checks that number
with no accumulated slack in between; with a per-level sum and the
simplifier's term removed it had failed (measured 0.034239, recorded 0.033778).

A level that removes less than one seventh of its triangles is rejected and
its children become roots. On the house this removes four levels that held
19,847, 17,327, 16,887 and 16,801 triangles at one and the same error.

## Results

`DamagedHelmet.glb`, `--hierarchy-levels 8`, two cooks byte-identical:

- structure unchanged from v3: 247 leaf clusters, 385 parents, 100 roots at
  level 3, 4,510 root triangles, 469,360 root page bytes, 2,937,040 payload
  bytes;
- errors by level: 0.0326 to 0.0545, 0.1945, 0.3393.

The house, `--hierarchy-levels 8`:

| | v3 | v4 |
|---|---:|---:|
| Levels above the leaves | 7 | 3 |
| Clusters | 5,173 | 2,992 |
| Archive | 24.0 MB | 14.1 MB |
| Error of the coarsest level | 6.300 m | 0.297 m |
| Hierarchy selection, one instance in view, M1 Max, 3200x1800, 1 px | 13.9 ms | 3.6 ms |
| The same with the allowed error at 16 px / 64 px | not measured | 2.1 / 0.5 ms |

A three-level v3 cook of the same house took 3.7 ms of selection at allowed
errors of 1, 8 and 64 pixels alike: with v3 errors the setting did not change
the work.

Rendered through the virtual path at allowed errors of 1, 4, 16 and 64
pixels, the v4 house keeps its texture in place at every setting; at 64 it
loses thin parts (pergola slats) first.

## The 10M stress gate needed a new source

The gate's plates (source v1) had per-vertex hash noise as their only relief.
Measured against the source, every level above the second then carries the
same error, the noise's amplitude (0.086 to 0.094 for levels 3 to 7): at the
gate's camera a one-pixel target is met by the pinned roots alone and nothing
streams, while a target low enough to refine at all asks for five levels at
once (at 0.3 px the selector took 5.3 ms against its 3 ms budget; at 0.5 px
there were no page reads). Source v2 adds waves at three scales, so each
coarser level loses a little more shape and a view needs detail gradually.
The target stays at one pixel.

| Gate, Metal, M1 Max | v3 cook, source v1 (`main`) | v4 cook, source v2 |
|---|---:|---:|
| Resident pages / I/O requests | 955 / 370 | 406 / 252 |
| Selected clusters, 100 instances | 10,220 | 4,900 |
| Selector GPU, 1 / 10 / 100 instances | 1.52 / 1.39 / 2.34 ms | 0.40 / 0.47 / 0.79 ms |
| GPU frame mean / p95 | 4.81 / 6.99 ms | 2.11 / 4.56 ms |

The v4 column was measured with the breadth-first selection and the
uniform-group shortcut in place; the two columns are different scenes and do
not compare the cooks.

## Tried and not adopted: collapsing across attribute seams

With meshoptimizer's `Permissive` option the house simplifies to 476
triangles at level 8 instead of stalling near 17,000. A triangle that spans
two UV islands then samples unrelated texels: patches of wrong texture were
visible at an allowed error of 4 pixels and obvious at 16. Position error
cannot see this. It needs a measured texture term (how far the texture moves
on the surface, bounded by the triangle's size) before it can be enabled.

## Qualification

- `cargo test --release` for `bloom-cook`: 53 tests pass, including the
  hierarchy oracle and a new test for measuring separate points against a
  surface; strict Clippy and `cargo fmt --check` clean.
- `ci-check.sh --hardware --component virtual-geometry-stress` on Metal
  passes; `tools/quality/test_prepare_virtual_geometry_stress.py` passes.
- Not re-run for this checkpoint: the Bistro runtime corpus (its asset is not
  on the machine this was written on). Its thresholds were tuned against v3
  archives and should be re-qualified with v4 cooks.
