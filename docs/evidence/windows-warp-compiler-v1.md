# Hosted Windows software-renderer compiler

The actual Windows shared job crashes with `0xc0000005` on #162 and #163.
The native engine build and all 20 example links pass. The Windows crash occurs
in the library stage, before golden tests execute; successful build jobs do not
qualify those render tests.

## Local diagnosis

At unchanged #163 source `c643b85`, four-thread library execution also crashes
when the virtual-geometry helpers select the Microsoft Basic Render Driver.
An eight-thread run passes, so a passing rerun does not close the defect.
Windows Event 1000 identifies `d3d10warp.dll` version `10.0.26100.9278`.

The installed Windows debugger captures a second-chance access violation in
`d3d10warp!JITBaseVariable::OptimizeCopy`, through the WARP DXIL shader-JIT
destructor and compute-pipeline compilation, during a traversal queue submission.
The affected Rust test refines atomic groups that straddle frustum planes.
This identifies the failing driver/compiler path; it does not establish the
underlying driver's root cause or prove that every hosted crash has this stack.

That existing test passes twice alone with DXC and twice with FXC. The complete
ten-test hierarchy group passes serially with DXC, crashes in both four-thread
DXC controls, and passes with FXC and four threads. Complete library controls with four and eight threads also
pass with FXC selected for environment-aware helpers (489 passed, one ignored).
Older helpers ignored compiler/backend environment options during those controls;
some explicitly select software adapters and others select the physical Radeon.

## Compatibility change

The hosted Windows shared lane explicitly selects DX12 and FXC. The remaining
ordinary GPU test helpers now honor backend and compiler environment options,
matching the visibility, virtual-geometry, and golden helpers. Adapter selection,
capability requirements, ignored tests, assertions and thresholds are unchanged.
The ray-query golden helper retains its deliberate DXC requirement.

This is a software-renderer CI workaround. The WARP/DXIL access violation remains
open, and physical DX12/DXC qualification remains required. The ordinary Windows
engine compiler policy is unchanged. The candidate's library suite passes with
four threads and explicit FXC (489 passed, one ignored). Complete local shared
components and hosted validation are still running.

Diagnostic commands, logs, debugger stacks, executable hashes and results are
retained in `tools/quality/out/windows-engine-plan/windows-shared-crash/`.
