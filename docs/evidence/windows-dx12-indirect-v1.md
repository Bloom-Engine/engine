# Windows DX12 draw identity and visibility oracle

Executing the Windows shared tests exposed two failures hidden by the previous
CI shell setup. Both reproduce on AMD Radeon 760M DX12 and Microsoft WARP,
with either FXC or DXC. The original full pixel checks pass under Vulkan.

## Counted indirect drawing

The GPU-generated commands contain the correct first-instance values, but the
DX12 counted path renders only the first identity. Submitting the same commands
through ordinary multi-draw or individual indirect calls passes the original
four-pixel identity oracle. Inspection of wgpu-hal 29.0.1 shows that counted
submission uses the common command signature, while ordinary indirect submission
uses the pipeline signature carrying the special first-vertex/instance constants.
The 29.0.4 implementation retains that count path.

Bloom now uses its existing bounded ordinary multi-draw fallback for DX12.
One device/backend policy governs static GPU-driven submission, virtual draw
emission, and virtual visibility rasterization. Reported counted-submission
support reflects that policy. Vulkan retains counted submission when enabled;
Metal retains its existing fallback. This is a correctness workaround; no DX12
performance improvement or upstream HAL fix is claimed.

## Visibility reconstruction oracle

A reduced shader isolates pipeline error `0x80070057` to the combination of
fragment `position` and `primitive_index`. Each input independently creates a
valid pipeline. This fails with either compiler on both DX12 adapters. The
production visibility ID raster does not consume fragment position; its
reconstruction runs separately.

The raster oracle carries linearly interpolated NDC and recovers the exact
pixel center before calling the unchanged production barycentric function.
Recovering the center is necessary because interpolation adds subpixel error;
a direct interpolated-coordinate control fails the original limits even under
Vulkan. Draw/primitive identity, opposite face orientation, clear sentinel,
nonuniform clip W, and the original `2e-5` barycentric limit remain checked.
No pipeline-only diagnostic, alternate shader injection, or unconditional test
success is retained in the source change.

## Validation and limits

Both focused original GPU oracles pass in five configurations: Radeon Vulkan,
Radeon DX12/FXC, Radeon DX12/DXC, WARP/FXC, and WARP/DXC (10 executions).
Adapter identities and successful execution are checked in the retained logs.
The affected test helpers now honor `WGPU_BACKEND`, shader compiler options,
and the test-only `BLOOM_TEST_FORCE_FALLBACK_ADAPTER` selector.

The initial full library runs pass on DX12 and Vulkan: 488 passed, zero failed,
one ignored. For the DX12 run, a process-local Vulkan loader override prevents
older test helpers using `Backends::all` from selecting Vulkan. All 32 emitted
adapter identities are DX12. No persistent graphics-driver setting changes.
The final patch also exercises the production virtual-visibility path with its
default submission choice as well as the explicitly forced binned path.
At committed source `ab8019b` in draft PR #162, the complete Vulkan shared CI
component passes: 489 library tests, device negotiation, 93 goldens, and the
remaining integration checks. The expanded DX12 run passes 489 library tests
and device negotiation, then reports 92 goldens passed, one failed, and four
ignored. The failed camera-cut equality check also fails on the unchanged
parent; its [history-read correction](windows-camera-history-v1.md) is separate.
Hosted Tests run 34547668613 still crashes in the Windows library process with
STATUS_ACCESS_VIOLATION and Bash exit 139 before completing the suite. The earlier hosted
access violation remains unresolved; subsequent non-crashing executions do not
establish a fix. Unsupported adapter features and missing optional asset fixtures
remain explicit limitations, and no unavailable hardware acceptance is claimed.

Diagnostic controls, original failures, candidate patches, commands, and results
are retained in `tools/quality/out/windows-engine-plan/hosted-windows-gpu/`.
The [native example build evidence](windows-example-gate-v1.md) is published
separately and does not qualify executable startup or clean installation.

The [published #162 report and ZIP](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-dx12-indirect-20260911)
retain 201 verified payloads, successful local/hosted native build receipts, and
the Windows access violation. The archive is 2,006,103 bytes with SHA-256
`ff50b4653062f2070508b825fc47f032e5907d5a054285e193ff33d2cb74975e`.
