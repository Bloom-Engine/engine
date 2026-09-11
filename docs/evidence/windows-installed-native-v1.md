# Installed Windows native package

The published Jolt dependency uses `lib/win32-x64`, while the engine's build
script searched for `windows-x64`. A clean install therefore ignored its shipped
archives and fell back to CMake. The long audit-project path then exceeded
MSBuild's path limit. The resolver now uses the published spelling and retains
the older spelling for staged CI archives.

After that correction, Cargo builds the engine but Perry's final link still
requests `Jolt.lib` and `bloom_jolt.lib` from a CMake output directory that does
not exist for a prebuilt install. Rust already bundles those static archives in
`bloom_windows.lib`. Removing the redundant Windows manifest libraries and
source-build `libDirs` allows the installed game to link. A live Jolt simulation
in the startup fixture verifies that the physics implementation is present.

## Local validation

The diagnostic installed-package candidate links and starts on physical Radeon
DX12 and Vulkan. Its 128x128 frame matches all 16,384 expected pixels on each
backend. The fixture creates a dynamic Jolt body, advances it deterministically,
and draws the expected white square only while the valid body has fallen under
gravity. It destroys the body, shape and world after the native loop exits.

The required Windows native-package check now packs and installs the actual
package into a fresh temporary project, compiles that fixture with the pinned
Perry toolchain, starts the native renderer and checks every frame pixel. It
rejects a CMake fallback, compiler/runtime errors, a missing capture and an
incorrect frame. Logs, source/package/compiler/archive hashes and captures are
retained; the temporary package, libraries and executable are removed.

The fresh complete checker passes using Jolt 0.4.1 with no CMake fallback. Its
native compile takes 267.812 seconds; DX12 and Vulkan startup/capture take 6.328
and 3.375 seconds. Both PNG files have SHA-256
`8a509d87d3aa3fab96e0a9e2c67228e187f1bc0853cb4726aa5844799440f30e`.
These are compilation/test durations, not a frame-rate measurement. The tested
installed build script, manifest and fixture match the candidate source.

Blank-frame and physics-failure-color controls are rejected. The full contracts
component, formatting and strict Clippy pass; Clippy also exercises the legacy
`windows-x64` staged-archive lookup. Hosted validation is pending. The runs use
the previously qualified Perry 0.5.1220 source/runtime profile. Local DX12 uses
SDK compiler DLLs from PATH. These results do not claim
stock-Perry usability, packaged DXC, a clean machine or window presentation.

## Separate limitations retained

- Supplying an external `CARGO_TARGET_DIR` lets Cargo build successfully but
  Perry then searches its usual crate-local directory for the engine library.
  That audit harness override was removed before the normal startup checks.
- Very long native project paths can still fail in MSVC's build-script linker;
  the resolver correction does not fix general Windows path-length support.
- The initial direct-2D probe exited without its queued PNG and failed. The
  accepted fixture in this report exercises the normal scene path. A subsequent
  [direct-frame capture correction](windows-direct-frame-capture-v1.md) adds
  both rendering modes to the installed-package check.
- Native headless startup does not prove browser startup or visible native
  presentation. #142/#74's complete starter and lifecycle remain open.

Raw diagnostics and successive candidates are retained under
`tools/quality/out/windows-engine-plan/installed-native-startup/`. Earlier failures
are preserved with the changes in harness configuration and source clearly marked.
