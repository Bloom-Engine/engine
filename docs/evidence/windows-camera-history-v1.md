# Camera cuts must not consume the previous history epoch

The expanded DX12 shared run exposed a failed exact-equality camera-cut check.
The failure also reproduces on unchanged parent source `59244b9`. With DXC,
one blue byte differs at pixel (179, 152): fresh 169 versus reset 168. FXC and
Vulkan pass the original eight-frame setup.

Extending the same test to a fully settled 40-frame history exposes visible
residue across backends. The unchanged shader differs at 16,888 pixels under
FXC and 16,893 under Vulkan, with maximum channel error 232. The original zero
tolerance remains required; the longer setup exercises a state the shorter
confidence ramp did not reach.

## Cause and correction

The CPU correctly marks reset and bootstrap frames as current-only. The shader
still sampled old color, indirect-light weight, depth, confidence, and detail
provenance whenever reprojected UVs were in bounds. Its existing history-usable
predicate was applied later. Stale confidence could activate settled-history
locks, and a final blend weight of one did not make old color arithmetically
irrelevant under every shader compiler.

Move the unchanged history-usable predicate to the read itself. Unusable history
now leaves the existing current-frame defaults and zero confidence in place.
Valid accumulation retains its previous predicate and policy. No texture clear,
new allocation, image baseline, or threshold change is introduced.

The permanent regression runs both eight and 40 old-history frames through the
same camera-cut equality check, then retains its projection/rotation/pan checks.
The golden device helper now honors backend compiler environment options.
Temporary capture instrumentation is retained only in the evidence patches.

## Verified controls

With the read guard, both history lengths produce byte-identical fresh/reset
RGBA on Radeon Vulkan, DX12/FXC, and DX12/DXC. All six captured comparisons have
zero differing pixels and zero channel error. Each backend's initial fresh image
also remains byte-identical to its corresponding original-shader fresh image.
The unchanged-shader controls retain their failures and images.

`Dx12Compiler::default()` in wgpu 29 is Auto, which tries DXC before FXC. The
original golden helper did not apply WGPU_DX12_COMPILER explicitly; with the
SDK DLL on PATH its parent-source control may select DXC automatically. The
isolated compiler matrix adds explicit backend options and distinguishes FXC
from DXC. The existing eight-frame FXC result is not a claim about Auto.

The complete local shared CI component passes under DX12/DXC and Vulkan:
489 library tests, device negotiation, all 93 golden render tests, and the
remaining integration checks pass. The existing one ignored library test, four
ignored goldens, and two ignored documentation examples retain their status.
Formatting, strict Clippy policy, and repository contracts also pass. The DX12
run uses a process-local Vulkan loader override for older helper functions;
some of those helpers explicitly use software adapters. The golden captures
require a physical GPU and identify the Radeon/backend. Hosted #163 passes the
macOS shared suite, including all 93 goldens and the expanded camera-cut test,
plus canonical Metal captures. Windows native builds and all 20 example links
pass; its shared library stage still crashes before goldens. That access
violation is separate and remains unresolved. This advances #135/#149/#140 without closing the wider
performance, platform, representative-corpus, or hardware requirements.

Source patches, commands, exact RGBA/PNG captures, changed-pixel coordinates,
and results are retained under
`tools/quality/out/windows-engine-plan/camera-cut-dx12/`.
The [published archive](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-camera-history-20260911)
includes exact source-match receipts and the failed Windows hosted log.
