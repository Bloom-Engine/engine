# Compiled-game browser startup acceptance

The existing browser gate calls the engine WASM directly from JavaScript. It
qualifies rendering and temporal reconstruction but cannot catch a trap in
Perry's game entry, FFI calls or closure dispatch as reported in #74.

The new gate compiles a real TypeScript game and an intentional startup-failure
control with the pinned Perry 0.5.1220 toolchain already prepared by Windows CI.
Both pages use the production splicer and bootstrap. Their compiler, entry,
embedded game WASM and assembled HTML hashes are retained. The browser job uses
the Linux-built engine package and the compiled pages at the same source commit
in hosted macOS Chrome, which has a working WebGPU presentation path.

The game initializes the window, selects direct-2D mode, draws a white square
on black for eight frames, requests a stop and records exactly one cleanup.
Acceptance requires the actual game progress and cleanup markers, no script
errors, and all 16,384 expected pixels in a 128x128 browser screenshot. The
monitor observes errors and adapter identity; it does not draw the test image
or call the game's update itself.

The failure control records entry through a real compiled `writeFile` call.
The monitor then injects a `WebAssembly.RuntimeError` at that FFI boundary.
It must produce that specific script error without successful frame or cleanup markers.
An unrelated infrastructure failure cannot satisfy that control. Unit controls
also reject missing progress, duplicate cleanup and an error-free fault marker.

The browser job now waits for both the engine build and the Windows fixture
compiler. This reuses the already qualified compiler instead of introducing a
second platform compiler/runtime download. Existing JavaScript-driven renderer
and temporal checks remain required and unchanged.

Both actual pages compile and splice locally, and acceptance-state controls
pass. Local browser execution is unavailable in this session; hosted execution
is the required rendering proof. No compiled-game browser result is claimed
until that check passes. Browser physics, the complete example runtime matrix,
visible native presentation, fixed-update lifecycle, one-command creation and
named hardware qualification remain separate.

The initial hosted Windows shared-test job at `1c147ae` exits with an access
violation despite the explicit FXC setting. That process failure is retained;
the earlier compiler workaround does not eliminate the instability. The next
CI attempt serializes the Windows Rust test harness, retaining all tests and
assertions. A local serial DX12/FXC library run at the same source passes 489
tests with one existing ignored test in 264.59 test seconds (333.515 seconds
including compilation). Selectable helpers use WARP; other helpers can use the
Radeon adapter. This local result does not prove every test on WARP or identify
the hosted crash's cause. The hosted serial result is still required.

## Initial browser failure and corrected compilation

The first hosted browser attempt passes the existing JavaScript renderer check
but times out after 90 seconds with no compiled-game frame or cleanup marker.
Its compiler had warned that both engine imports were unresolved, yet exited
zero and emitted a 10,870-byte WASM module. The original local compilation had
the same warning. Those outputs are invalid game qualification, despite their
valid WASM headers. The new preparer creates a project with an explicit local
engine dependency, verifies it resolves to the exact checkout, rejects unresolved
import warnings and requires the intended engine FFI imports in the actual WASM.

The corrected local fixture compiles six modules and includes 159 engine imports.
A Node VM probe executes the generated Perry runtime and game WASM against a
recording FFI, verifying eight callbacks, one cleanup and the fault control.
It does not load Bloom's renderer or replace hosted browser acceptance.

That probe also found Perry 0.5.1220's plain TypeScript throw only sets internal
exception state in this fixture and continues into game initialization. The
control now explicitly injects a runtime error at the compiled FFI call, which
the actual Perry boot promise rejects. General Perry exception propagation
remains a separate limitation; this change does not claim to fix it. Failure
diagnostics now include startup logs, DOM status, FFI count and a screenshot.
