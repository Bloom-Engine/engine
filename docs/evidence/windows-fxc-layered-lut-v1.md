# FXC layered-material lookup sampling

The expanded physical Radeon DX12/FXC run on #164 passes 89 rendering goldens
and fails four during `scene_layered_pbr_pipeline` compilation:

- clearcoat-normal minification and motion;
- explicit mirrored-tangent anisotropy;
- anisotropy under negative model scale;
- layered materials across opaque, sorted, reactive and weighted paths.

FXC reports X3570 for a gradient instruction in a varying loop, followed by
X3511 when it cannot unroll the loop. The shader, scene specialization, core
shader, test definitions and golden helper are byte-identical between #163 and
#164. The source-equivalence receipt distinguishes that comparison from a
separately executed parent full-suite run. These are compiler failures before
image assertions, and their original logs remain in the #164 evidence.

## Correction

`layered_sheen_directional_albedo` used an implicit-derivative texture sample.
The helper is called from varying direct-light loops. Its R16Float lookup table
has exactly one mip, so explicitly sample level zero with the same texture,
sampler and lookup coordinates. This removes the implicit gradient requirement
that forced FXC to unroll the surrounding loop. No additional binding, texture,
allocation, feature requirement or compiler fallback is introduced.

The change applies to the shared layered-material helper, including its scene,
transparency and visibility uses. The existing sampler policy remains intact;
the successful image assertions below are the evidence for rendering behavior,
not an assumption that compiler outputs are byte-identical.

## Validation

The complete local DX12/FXC shared component passes: 489 library tests, device
negotiation, all 93 goldens and the remaining integration checks. All four
previously failing tests now create their pipelines and pass their original
rendering assertions. Existing one ignored library test, four ignored goldens
and two ignored documentation examples remain unchanged.

Raster goldens require and identify the Radeon 760M/DX12 adapter. Ordinary
pipelines use explicit FXC; the ray-query golden helper deliberately requires
DXC. Older library helpers can explicitly select software adapters. The full
FXC component takes 1,073.406 seconds including compilation, shader creation and
tests; this is test duration, not an FPS or GPU-budget measurement.

DX12/DXC and Vulkan each pass all 93 goldens with four existing ignored tests.
Strict Clippy, formatting and repository contracts also pass. The test driver
records the exact candidate shader patch against `f96af2e`; only report updates
followed these runs. Hosted validation is pending.
The WARP/DXIL concurrent compiler crash documented with #164 remains a separate
open defect. This change does not qualify unavailable named hardware or wider
performance, starter, packaging or engine-plan acceptance.

Source patches, the original failure log and source-equivalence receipts,
commands and complete results are retained under
`tools/quality/out/windows-engine-plan/fxc-layered-lut/` and
`tools/quality/out/windows-engine-plan/windows-shared-crash/`.
