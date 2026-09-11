# Public constants and portable example browser acceptance

The six shared-loop examples compile for WASM, but executing their original
game code found undefined RGBA arguments in every example. Reading the palette
implementation directly returns the correct values; importing it through the
public `bloom/core` barrel returns undefined bindings in Perry 0.5.1220 WASM.
The same probe finds undefined public `Key` and `MouseButton` values. Package-root
imports also lose palette, input, platform, cursor, texture-kind/filter and
material-slot constants. Native and WASM compilation alone do not prove those
public bindings are usable.

Explicit initialized exports in the two barrels preserve the original names,
values, TypeScript types and shared object identities. The palette definitions
and numeric key values do not change. Actual native/WASM execution now verifies
all 24 colors through both naming styles and both public entry points, shared
color mutation/restoration, 102 input/platform/cursor values through each entry
point, and the ten package-root texture/material constants. Its console runner
can admit unused FFI imports only with guards that reject every engine call.
A separately compiled control calls `getPlatform()` and must fail at that guard.
Default pure timing/scalar fixtures still reject FFI imports entirely.

The expanded browser preparer copies each original example byte-for-byte into
an explicit local dependency project, compiles it with the pinned Perry binary,
verifies the real WASM imports and splices the production bootstrap. Source and
page hashes survive the artifact transfer. Hosted Chrome uses the qualified
engine WASM package, normal physics bootstrap and the original game pages.

The monitor observes actual FFI and Perry callback dispatch, then requests normal
engine stop after eight updates. It requires one completed cleanup, the intended
viewport and drawing in every frame. Undefined/non-finite drawing arguments and
missing key/mouse codes are failures. Space-blaster must initialize and close
audio; voxel-sandbox must restore cursor intent. Captures must satisfy the same
game-content checks as the native examples. The monitor does not replace game
draws, supply input values or inject a fake physics factory. Browser evidence runs
even if a later Windows native check fails; absent or invalid compiled artifacts
still fail, and the native job remains required.

Local recording-FFI execution of the actual compiled games rejects all six
earlier pages and accepts all six after the shared binding correction. This
diagnostic uses the production scheduler, but does not load a browser, renderer,
GPU, audio device or network. The first local probe selected the wrong boot
script and failed before game execution; that setup failure is retained. The
audio observer was also corrected to use the actual `bloom_init_audio` and
`bloom_close_audio` names before hosted qualification.

With the shared bindings corrected, all six native content/capture/cleanup
checks pass on Radeon 760M / DX12, and all 20 canonical examples compile/link.
The unchanged six WASM pages pass the final recording-FFI state verifier,
including audio and cursor cleanup. Actual fixed lifecycle and scalar math
contracts still pass in native/WASM execution. Fourteen Python acceptance
controls, four monitor tests and the repository contracts pass. The first
repository contract attempt failed because the new worktree had no initialized
Jolt submodule; it passes after checking out the existing pinned submodule.

The initial #174 hosted run, 34576806162, passes the scalar contract and all
20 example links. All six native executions then fail with `TypeError: Expected
number for native f64 parameter`, after WARP adapter initialization and before
capture/cleanup. Its logs and reports remain retained. Local native success does
not replace this failed hosted requirement; the expanded public binding
correction and runtime gates require a fresh hosted run.

The corrected source still needs actual hosted example browser acceptance.
Eight startup frames do not qualify full gameplay or input interaction, pause/
focus policy, resize/device loss, quality goldens or performance. The missing
test3d grid lines, broad generated API/export audit and general compiler
qualification remain open. No discrete-GPU acceptance is inferred from hosted
Metal or Windows software rendering.
