# Fixed game lifecycle

The callback-only loop supplies variable delta and leaves timing and hook order
to each game. The new optional `runGameLifecycle` entry defines init, fixed
update, variable update, interpolated draw and one-time cleanup for native and
web. `FixedStepClock` provides bounded catch-up and explicit dropped-time
diagnostics. The original `runGame` API stays supported.

The pure TypeScript contract is compiled and executed with the qualified Perry
0.5.1220 profile both as a Windows executable and as actual WASM in its generated
runtime. Uniform and varied partitions each produce 50 ticks; a 6,000-frame
144 Hz sequence produces 2,500 ticks at 60 Hz. Checks cover catch-up/drop and
resume, interpolation, invalid settings/deltas, overflowing accumulation, very
small steps, optional hooks, initialization/disposal order, duplicate cleanup,
stop during fixed update and stop during variable update. Host-side controls
reject missing, duplicate, malformed and incorrect observations.

Initial failed runs are retained. Object-literal JSON reporting returned
undefined in WASM, so the fixture now reports the actual scalar observations
directly. `Number.isFinite` has a distinct HIR expression without a corresponding
WASM lowering in pinned Perry; an attempted runtime dispatch shim did not fix
that and was removed. A portable noncoercing arithmetic finite check now serves
the clock and existing quality/scene validators. Method syntax for stored
callbacks also went through named dispatch instead of the closure bridge.
Reading each callback into a local function reference fixes the native/WASM
observations without changing Perry. General JSON/exception compatibility is
not claimed.

A fresh installed candidate package compiles the public lifecycle fixture and
renders its exact 128x128 image on Radeon DX12 and Vulkan. Both captures match
all 16,384 pixels and PNG SHA-256
`8a509d87d3aa3fab96e0a9e2c67228e187f1bc0853cb4726aa5844799440f30e`.
Each records one init, nine updates/draws, two fixed ticks and one cleanup, with
valid interpolation. The fixture captures frame eight and exits after readback.
This lifecycle mode does not simulate Jolt; the existing scene/direct-2D modes
retain their independent physics acceptance. An older generic success message
in the first local log mentions Jolt; the mode-specific observations establish
the actual scope, and the harness message is corrected.

The same fixture compiles for web and a recording-FFI probe verifies all hooks,
13 fixed ticks across eight supplied 1/60-second frames, interpolation and the
explicit startup-fault control. That probe uses real Perry WASM/runtime but no
renderer. Hosted browser rendering of this revised fixture remains required.
The Windows job now runs the pure native/WASM contract and retains its evidence;
the installed native gate adds this lifecycle mode to its existing two modes.
The browser gate requires lifecycle counters in addition to the exact image.

The starter uses fixed simulation and previous/current interpolation. A fresh
installed CLI creates it with the default command and builds the unmodified
native template in 266.563 seconds. `npm start` builds/runs a bounded copy in
7.047 seconds: the 800x450 image shows the greeting and interpolated square,
the text asset loads, and cleanup runs once. Restoring the unmodified source
completes the full web build with its asset manifest. Seven installed runtime
and template sources match the candidate. This is local headless native and
web-build acceptance; full starter browser rendering is still unperformed.
Repository contracts and eight acceptance-control tests pass. Broader examples,
components, input/pause policy,
device-loss recovery, full starter browser assets/text and named-hardware
qualification remain separate. No issue is closed by these focused checks.
