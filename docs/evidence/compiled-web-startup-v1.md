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

The failure control records entry into its intentional fault and then throws.
It must produce a script error without successful frame or cleanup markers.
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
