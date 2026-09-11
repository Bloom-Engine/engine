# Installed starter command acceptance

The new package includes `bloom new/run/build`, a shared TypeScript template,
asset example and local web server. The generator carries the exact engine
archive into each project, avoiding a preview/stable collision at the existing
0.4.16 version number. Native build/run selects the host platform; web uses the
existing actual Perry-plus-engine builder and writes a prefetch manifest.

## Local execution

The first fresh-package probe uses the explicit local archive option. Packing,
bootstrap install, installed command help and installed project creation all
pass. The unmodified starter builds natively in 270.360 seconds. The installed
`npm start` path then builds/runs a bounded copy in 10.141 seconds. Capture/state
instrumentation preserves the template's init/update/draw/cleanup logic.

The 800x450 capture shows the greeting loaded from `assets/welcome.txt` and the
moving white square on black. Checks require its observed square center,
background corners, at least 4,096 white pixels and exactly one cleanup. Visual
inspection also confirms the greeting. The PNG SHA-256 is
`be7a32be2fb93fb24fa838370dc9c656c7b363ba19fe9b12f1681aa4b9d1acbd`.
These checks are not a claim that all 360,000 pixels are an exact golden.

The probe restores the unmodified source and completes the full web build in
63.765 seconds. The generated asset manifest includes `assets/welcome.txt`,
whose bytes match the source. Engine WASM SHA-256 is
`c81e7d5760d7fc4de30b14f8a177f71b1a0988669252eab7e7e1aaedc7342d65`.
No compiled-starter browser rendering is claimed by this build result.

The local native run uses the qualified Perry 0.5.1220 source/runtime profile,
Radeon 760M DX12, headless pixel-exact output and SDK shader DLLs on PATH. It
installs Jolt from the package dependency without repository prebuilt overrides.
Executables and the installed diagnostic project remain private.

## Creation and failure controls

The default creation follow-up packs the running installed engine instead of
selecting an older registry package with the same version. The installed-package
checker verifies the generated source, packaged command/setup files, asset and
archive reference, and requires nonzero actionable failures for a missing
compiler and unsupported target. This local default-creation check passes.
The cached managed Windows profile also passes compiler/library hash validation
and selects its matching runtime environment. It does not exercise a fresh
toolchain download. Hosted acceptance remains pending.

Five unit checks cover help without tools, argument rejection, manifest drift,
existing-project preservation, explicit archive copying, compiler/process
failure causes and local serving. Repository contracts pass. Local HTTP checks
request an owned test server's generated files; they do not control a browser.

Visible native startup, actual compiled-starter browser rendering, macOS/Linux
starter runtime, automatic fresh-machine toolchain setup through this CLI,
all-example runtime coverage, fixed updates, hot reload and distributable
shader-runtime packaging remain open. The tested native CLI uses an explicitly
prepared toolchain. No npm release or PR merge is performed.
