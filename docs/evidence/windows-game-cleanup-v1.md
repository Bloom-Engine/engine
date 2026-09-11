# Shared game cleanup and canonical example startup

`runGame(updateAndDraw, cleanup)` adds an optional cleanup callback without
changing existing callback-only calls. Native invokes cleanup after the normal
loop exits. The browser scheduler defers it until an active frame ends, cancels
future frames and invokes it once. Frame failures stop scheduling and report an
error; cleanup is attempted. The [public contract](../game-loop.md) records
timing, duplicate-start, focus and fatal-process limitations.

Pong now uses this loop, disposes audio/window resources in cleanup and toggles
pause with `isKeyPressed(Key.P)`. Its actual runtime check also found an earlier
example correction was incomplete: `Colors.White` and similar names are
undefined because `Colors` exposes uppercase aliases. All 51 such references
in six canonical examples now use valid names. The example inventory checks
references against the exported palette, with invalid mixed-case names as a
negative test. This catches a failure that Perry's native linking did not.

## Local acceptance

- A fresh actual installed package simulates Jolt, captures all 16,384 expected
  pixels and records exactly one cleanup call in both scene and direct-2D modes,
  on Radeon DX12 and Vulkan. All four PNGs are byte-identical. Native compiles
  take 270.609 and 3.390 seconds; these are test durations, not FPS.
- A separate real native callback-only probe still renders the same exact
  physics frame, returns from the loop and reaches its existing post-loop
  cleanup. The optional parameter does not break this older native pattern.
- Compiled Pong input replay records nine frames after staging a held P key.
  The old held-key control yields `RPRPRPRPR`; the edge-triggered version yields
  `RPPPPPPPP` (R=running, P=paused). The first frame precedes application of the
  queued input. Both variants start and close on Radeon DX12.
- All 20 native canonical examples compile and link in 134.884 seconds. This
  does not claim runtime acceptance for every example.
- The same corrected Pong source and actual engine complete the web build in
  72.875 seconds, with the new scheduler copied into the assembled distribution.
  The 7,829,878-byte engine WASM and generated bindings include the new bridge.
  Browser rendering is not claimed by compilation and assembly.
- Nine scheduler tests and nine web-command regression tests pass. The example
  gate rejects invalid palette names while accepting the public uppercase keys.

Repository contracts, the complete quality-contract component, formatting,
strict Clippy and the shared WASM compile check pass. The complete web build
also compiles the web crate and generates its new FFI export. Hosted validation
is retained with the final published evidence.

The initial Pong runtime failure and the intermediate pause-audit expectation
error remain recorded. That audit initially counted eight frames as eight held
updates; input staging makes it seven. The final replay records the full state
sequence instead of inferring it from an ambiguous last frame.

The runs use the qualified Perry 0.5.1220 source/runtime profile and local SDK
DXC on PATH. A fresh installed native project has its own Cargo outputs; the
canonical example audit reuses the native build cache. Native presentation,
packaged DXC, actual compiled-game browser startup, fixed updates, a complete
lifecycle and one-command creation remain separate requirements.

Raw diagnostics are retained under
`tools/quality/out/windows-engine-plan/starter-lifecycle/`. Test executables,
installed trees and generated HTML/WASM stay outside release-evidence archives.
