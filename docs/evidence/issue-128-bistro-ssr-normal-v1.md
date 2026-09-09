# Issue #128 Bistro SSR normal evidence

This record qualifies the SSR normal correction at revision
`56bf87c7c3def12cc2552c96c77a2209a4eb4790` with the governed
`bistro-exterior` case on Apple M1 Max / Metal. The case renders 96 unique
camera-visible Bistro meshes for 180 warm-up and 240 measured frames at
800x450, high quality, native render scale, TAA, SSR, SSGI hardware ray query,
bloom, fog, sun shafts, imported transparency, and transmission.

## Defect and correction

SSR reconstructed its derivative normal as `cross(dpdx, dpdy)`. Fragment Y
increases downward, so that normal faced away from the camera. `NdotV` became
zero over camera-visible geometry and Schlick Fresnel became 1.0, flooding the
frame with a full-strength pale environment reflection.

The shader now reverses the cross-product operands and faces the result toward
the view vector at depth discontinuities. The ordinary and layered SSR shader
variants share this source. A Naga parse/source-contract test prevents the
operand order from regressing.

The captured SSR mean luminance fell from `0.266009541` to `0.011289671`
(23.56x lower), with zero non-finite pixels and zero isolated local outliers.
The final image changed from SHA-256
`b39e19822372269c0ec17393de98643668c700e075872406ae38fad778fdf27c`
in the original governed run to
`3bfb6ae42cd8e2f86ceedbbbb53ab2352b7f91236b49aca7fb1894fa1573cbba`.
The corrected candidate preserves the green storefront, red brick, cobble
contrast, and authored material detail with the complete post stack enabled.

## Cost and gates

The SSR pass remained effectively flat: 2.639690 ms before and 2.643431 ms
after (+0.003741 ms, +0.14%). No pass, texture, buffer, bind group, graph
compile, or transient allocation was added. Steady state retained one compiled
25-pass graph with 419 cache hits and zero per-frame graph, pipeline, bind-group,
texture, or buffer creation.

The release renderer suite passes 357 unit tests, 59 runnable GPU goldens, four
render-target tests, and its device-construction test. Strict Clippy/format,
Wasm compilation, FFI/schema contracts, cooker/quality governance, and the
canonical example inventory also pass.

The governed run reports only the pre-existing missing human-approved Bistro
baseline. This candidate is evidence, not an automatically approved baseline.
