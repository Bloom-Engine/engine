# Issue #24 SSGI foreground revalidation v1

Bloom revision `6677c28c44e52bdfea5c36f937ee0bf539e2edd8` preserves the
accepted detailed-Bistro SSGI fixes after the subsequent visibility,
virtual-geometry, and asset-runtime work. Qualification was performed from a
clean detached worktree so unrelated local Three.js bridge changes could not
enter the renderer or executable.

## Result

The exact 1,176-placement `BistroReference.gltf` scene passed both the
deterministic temporal owner matrix and a fresh native foreground inspection.
The reviewer traversed bright areas and inspected the paving under camera
motion, then reported **“looking good all around!”**. No `N` failure capture
was requested. The old camera-following red/bright facade projection did not
return, and the remaining ground response was not objectionable in the
foreground pass.

The scene SHA-256 is
`217b6cf37bfb435eb6d164a6107e19f69f4bbf4d1bfb022713aa7076a8752927`.
The fresh executable SHA-256 is
`e7e8ba551364efe63b9188c78860af76ad2348e1c1abeddba80c3f64a82d6d8b`.
It was compiled with Perry 0.5.1512 and launched at 1600x900, render scale
1.0, quality preset 4, hardware-ray-query SSGI, TAA/SSGI/SSR on, and
sharpening/motion blur off.

## Deterministic temporal gate

The canonical five-variant matrix ran 32 stationary frames at 512x288 after
full admission and the matched camera excursion. The full path's largest
adjacent-frame component over eight 8-bit levels was **25 pixels**, passing
the permanent limit of 32 pixels. Mean accumulated max-channel range was
2.129517, mean adjacent RGB change was 0.138045, and the changed-pixel ratio
was 0.0015191.

Owner controls remain consistent with the accepted diagnosis:

| Control | Accumulated-range reduction | Adjacent-RGB reduction | Largest-component reduction |
|---|---:|---:|---:|
| TAA off | 94.20% | 98.11% | 92% |
| SSGI off | 4.44% | 2.41% | 0% |
| SSR off | -0.37% | -3.65% | -8% |
| Hi-Z occlusion off | -0.004% | 0.013% | 0% |

This excludes a return of the former SSGI transport leak: removing SSGI does
not remove the coherent event, while removing final TAA reconstruction nearly
eliminates it. The matrix JSON SHA-256 is
`23d08be2b7db20a0d12be64c1bbd002015ed0bbbef12e7d988ab8da3a8b543d5`.

## GPU performance

The accepted SSGI profiling checkpoint `b92668f23f45b076660af56a92e4d72ab9c1599f`
and current revision were built in separate clean worktrees and run in five
alternating release pairs on the same Apple M1 Max / Metal device.

The historical 256x256 Hi-Z fixture reported a median four-probe-pass total of
682.258 microseconds for the baseline and 662.301 microseconds for current
HEAD: **-19.957 microseconds / -2.93%**. All ten captures had byte-identical
quality metrics between revisions: 14,016 current probes, 14,592 retained
probes, zero non-finite pixels, max luminance 0.194479, and stationary TAA
SSGI SSIM 0.9887517366.

The fully admitted detailed-Bistro endpoint profiler independently reported a
median total of 6,371.515 microseconds for the baseline and 6,266.866
microseconds for current HEAD: **-104.649 microseconds / -1.64%**. This
includes place, hardware trace, temporal, spatial, and resolve work at both the
moved and direct endpoints. All ten endpoint tests passed.

An exploratory `BLOOM_SSGI_PROFILE_HD=1` invocation was not used as evidence:
that override changes the small Hi-Z fixture's quality regime and reaches its
stationary phase-stability assertion before timing output at both revisions
(identical SSIM 0.982474). The accepted default-resolution fixture and the
real detailed-Bistro profiler are the valid matched comparisons above.

## Regression suites

- release shared library: 482 passed, zero failed, one intentional ignore;
- release GPU golden corpus: 79 passed, zero failed, two hardware-only ignores;
- deterministic five-variant Bistro matrix: passed;
- five alternating synthetic timing pairs: 10/10 passed;
- five alternating detailed-Bistro endpoint timing pairs: 10/10 passed;
- native foreground bright-area/paving camera sweep: passed.

No renderer change is required by this revalidation. Revision `6677c28` is a
stable pushed renderer checkpoint; this evidence commit records that the SSGI
world-surface ownership and moving reconstruction fixes remain intact through
the later renderer work.
