# Native example build gate

All 20 canonical examples compile and link on Windows with the same Perry
0.5.1220 compiler and matching source-built runtime profile. This includes
`perry-embed`, whose native-view API is absent from the earlier 0.5.1182 release.
The gate retains a fresh executable's byte count and SHA-256, compiler command,
environment, and logs for every example. This establishes native linking;
startup and rendered-frame acceptance remain separate.

## Toolchain findings

The 0.5.1219 and 0.5.1220 Windows bundles have unresolved HTTP-extension symbols
in their prebuilt full standard library. Rebuilding the exact 0.5.1220 source
at `06137858dc8c6f80975238377138f2f948d6ef88` resolves the embedded-view link.
Perry's automatic per-app profile then exposes five different link failures:
Dungeon Crawl, Isometric RPG, Pong, Space Blaster, and Test Scene Watch fail
with relocations into discarded Rust COMDAT sections when linking Bloom's
unwind library with panic-abort runtime archives. All other 15 examples link.

Using one unwind profile with the union of required runtime features resolves
those failures. The profile enables runtime `full` and `regex-engine`, and
stdlib `async-runtime` and `crypto`. The setup tool verifies the official ZIP
SHA-256, exact source commit, unchanged source, and Cargo lockfile; builds the
libraries with `--locked`; and records their hashes and Rust compiler identity.
It uses Perry's supported runtime-directory and auto-optimization overrides.
The compiler and its source are unmodified.

The initial all-example invocation with this profile passes 20/20 native links
in 157.672 seconds on this host. Build time is an observation, not a performance
budget. Earlier failed profiles and their logs are retained.

## Required checks

The quick lane links Pong. The full and hardware lanes link every canonical
example. Windows PR CI runs the same full-lane component after building the
engine, with a pinned source/toolchain cache and required execution summary.
The compiler gate rejects zero exit status without a new native binary, so an
old output or object-only compilation cannot count as success. Failures and
timeouts do not hide later example results.

Three orchestration regressions, the full quality-contract component, and
repository contracts pass locally. Hosted example execution is pending.

The preceding [Windows CI correction](windows-example-ci-v1.md) now proves an
actual hosted native build at `636b69a`. Its shared suite exposes two DX12 GPU
failures and a process crash. Both focused GPU failures also reproduce on the
Radeon and the Microsoft software adapter with explicit DX12 selection; Vulkan
passes. Those failures remain required and are under separate investigation.
No broader CI, startup, packaging, or hardware issue is closed by this report.

Commands, failed profiles, successful executable hashes, setup receipts, and
logs are retained under
`tools/quality/out/windows-engine-plan/all-examples/`.
