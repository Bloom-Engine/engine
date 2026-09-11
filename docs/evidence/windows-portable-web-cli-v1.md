# Portable installed web command

The actual npm package installed successfully in a clean Windows project, but
`npm exec -- bloom-web --help` failed because its generated command shim tried
to run `/bin/bash`. Invoking the script directly through Git Bash in repository
checks did not exercise that installed entry point.

The old build also piped `wasm-pack` into `tail` without `pipefail`, allowing a
failed engine compile to appear successful and reuse stale `pkg/` output.

## Change

The package now exposes a Node entry point. It passes executable arguments
directly, resolves game/output paths from the caller, compiles in a fresh
temporary directory and assembles the distribution only after successful
compilation and artifact checks. Missing tools, nonzero compiler exits, missing
outputs and incompatible Perry HTML stop the build. The optional optimizer may
be absent; a present or explicitly configured optimizer must succeed.

The existing shell and Python entry points delegate to the same Node build and
splicer implementations. Bash and Python are no longer package build
dependencies. Node 18 or newer is declared. The web guide distinguishes this
upcoming branch from the stable 0.4.16 npm command.

Nine regression checks cover the real subprocess exit boundary, stale/missing
artifacts, spaces and assets, repeated builds, optimizer failures and Perry
bootstrap validation. Their fake compiler outputs are orchestration fixtures,
not evidence of engine compilation or rendering. Repository contracts run them;
the Windows build job additionally packs and installs the package in a clean
project and invokes npm's actual installed command.

## Local verification

- All nine regression checks and the complete contracts component pass.
- The candidate package packs and installs with pinned Jolt dependency 0.4.1.
  Lifecycle scripts are disabled; both package manifests have no lifecycle scripts.
- The installed `npm exec -- bloom-web --help` passes on Windows.
- The installed command compiles a bounded TypeScript probe using Perry
  0.5.1220, builds the engine with wasm-pack and assembles a distribution in a
  path containing spaces. The full build passes in 107.140 seconds. Its engine
  WASM is 7,839,672 bytes; the asset copy and gated HTML bootstrap are checked.
- The Windows CI pack/install script also passes locally. An initial probe
  exposed Windows PowerShell's UTF-8 BOM in a generated package manifest; the
  script now writes that manifest without a BOM and retains native exit codes
  independently of stderr.

Commands, exact candidate patch, package and artifact hashes, full build logs,
and install receipts are retained under
`tools/quality/out/windows-engine-plan/portable-web-cli/`.

Hosted validation is pending. The local browser connection is unavailable, so
no browser frame or runtime startup result is claimed here. Native startup,
the one-command starter, shared lifecycle, wider example runtime matrix and
packaged shader-runtime acceptance remain open under #142, #74 and #145.
