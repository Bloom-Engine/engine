# Issue #131 complete Bistro runtime v2 evidence

This checkpoint qualifies Bloom's opt-in virtual-geometry path against the
complete `bistrox.gltf` scene at revision
`60249af72f6dfe7350bc37e5d5754b8aa4c1d086`. Unlike the earlier
`BistroReference.gltf` slice, this source includes the complete interior camera
path and every 2,909 mesh placement. The fixed motion gate exposed a
single-sided face-ownership mismatch that the outdoor-only path did not reach.

## Corrected contracts

Three source/runtime boundaries required correction before the complete asset
could be qualified:

1. Twelve tangent accessors contain 847 vertices whose optional tangent has
   non-finite XYZ components. Position, normal, UV, color, and index data are
   finite. The cooker and ordinary glTF importer now convert only a malformed
   optional tangent to Bloom's established all-zero missing-tangent sentinel.
   The cooker reports the sanitized vertex count, while all required geometry
   and every other optional attribute retain fail-closed validation.
2. Active `KHR_materials_transmission` and authored layered-PBR extensions are
   not owned by the current virtual material ABI. They now receive stable,
   inspectable `transmission` or `layered-pbr` compatibility reasons during
   cooking instead of reaching a late runtime material-binding failure.
3. Virtual visibility had stopped discarding back faces for single-sided
   opaque glTF materials. The ordinary retained-scene pipeline uses back-face
   culling. At motion steps 25 and 30 the virtual path therefore rendered the
   reverse side of an interior wall over the complete frame. Virtual
   visibility now discards a raster back face unless the cooked material is
   explicitly double-sided. Raster front-facing state already includes a
   mirrored instance's winding reversal.

The corrected late frames expose the same bar interior, railing, radiator, and
exterior wall as the ordinary renderer. Streaming remained deterministic; the
failure was face ownership, not missing pages, history, or an overflow.

## Workload and deterministic artifact

The source has 551 meshes/primitives, 2,909 placements, and 1,736,174 eligible
source triangles. Runtime routing assigns 2,404 placements to virtual geometry
and 505 to the ordinary compatibility renderer. The artifact records 45
incompatible source primitives: 42 active-transmission and three alpha-blend.
The cook sanitizes exactly 847 optional tangent vertices.

Two release cooks using `--vertex-format quantized32 --hierarchy-levels 8`
produced the same payload SHA-256:
`e379cdb4cc3210de2c053bc0726b6968e923b5530f6397706575587ee3b90e5f`.
The source-closure SHA-256 is
`89d1ea17e0850f0961b51f4d4265cc858879072b1a884a415ccc927d8cb0c49b`.

The strict version-2 artifact is 160,522,736 bytes. It contains 70,746
clusters in 2,349 pages, reaches LOD level 7, and pins 485 root pages consuming
31,291,696 physical-slot bytes. The useful payload is 151,316,032 bytes. The
reported maximum quantized UV absolute error is 8.98046875 for this heavily
tiled source; quantized virtual geometry remains explicit opt-in and this
qualification compares its accepted pixels directly against ordinary glTF.

## Fixed camera-motion gate

Ordinary and virtual children render separately at 640x360 with TAA, SSAO,
SSR, SSGI, bloom, motion blur, subsurface scattering, sharpen, auto exposure,
shadows, and sky disabled. The virtual child warms for 180 frames, traverses 30
camera steps, reverses the same path, settles for 30 frames, and captures the
identical start camera again. Captures at every fifth step are compared to the
ordinary child and to the matching return-path camera.

| Comparison | Mean RGB | SSIM | Missing geometry | Background leak |
|---|---:|---:|---:|---:|
| Start ordinary vs virtual | 1.170891 | 0.96349717 | 0% | 0% |
| Step 5 ordinary vs virtual | 0.243592 | 0.99632865 | 0% | 0% |
| Step 10 ordinary vs virtual | 0.536338 | 0.98290636 | 0% | 0% |
| Step 15 ordinary vs virtual | 0.639631 | 0.97444583 | 0% | 0% |
| Step 20 ordinary vs virtual | 0.687815 | 0.97672083 | 0% | 0% |
| Step 25 ordinary vs virtual | 0.943374 | 0.97004718 | 0% | 0% |
| Step 30 ordinary vs virtual | 0.472125 | 0.99083564 | 0.001736% | 0% |
| Virtual start vs returned start | 0 | 1.0 | 0% | 0% |

The enforced ordinary/virtual threshold is mean RGB at most 3, SSIM at least
0.95, missing geometry at most 0.5%, and background leak at most 0.1% for the
start and every sampled moving frame. Matched outbound/return cameras require
SSIM at least 0.985. The observed minimum path-return SSIM is 0.99999873 and
maximum path-return mean RGB is 0.00003617.

At the starting camera the Metal runtime held 1,204 pages in the fixed 128 MiB
physical pool, selected 1,806 clusters, refined 505 groups, and reported zero
fallback groups, missing-current pages, selected/request/page-use overflow,
invalid records, or depth-limit fallbacks. Every captured moving report also
reported zero missing-current pages and zero overflow.

## Automated qualification

- complete Bistro release parity/motion gate: pass;
- deterministic eight-level complete Bistro recook: pass;
- shared library with `models3d`: 481 passed, one existing ignored;
- cooker: 48 passed;
- geometry format: one passed;
- strict `-D warnings` Clippy for cooker and geometry format: pass;
- Rust formatting and diff whitespace: pass.

Repository-wide shared strict Clippy is not claimed by this checkpoint: the
current toolchain reports 188 existing lints across unrelated shared modules.
The complete shared test suite is the shipping regression gate used here.

## Remaining hardware qualification

This run qualifies the fixed complete-scene corpus on an Apple M1 Max
integrated GPU using Metal. Issue #131 explicitly requests captures on at least
one integrated and one discrete GPU. The camera-motion acceptance item therefore
remains open until the same enforced test passes on a discrete adapter/backend;
no threshold is weakened for that run.
