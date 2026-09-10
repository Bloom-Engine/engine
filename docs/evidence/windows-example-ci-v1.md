# Windows example linking and CI execution

A native compile-and-link audit of all 20 canonical examples at `d610d6a`
produced 13 successful Windows executables and seven failures with the pinned
official Perry 0.5.1182 toolchain. This exposed failures that an inventory check
or `--no-link` compilation cannot establish as passing native builds.

Six examples still accessed `Color.White`, `Color.Red`, and other palette values
after `bloom/core` stopped exporting `Color` as a value. `Color` remains the RGBA
type; `Colors` is the public palette. Correcting those imports and accesses
makes Pong, Dungeon Crawl, Isometric RPG, Kart Racer, Space Blaster, and Voxel
Sandbox compile and link on the same compiler. Type annotations retain `Color`.
No renderer or library API changed.

The remaining `perry-embed` failure is independent. Perry 0.5.1182 does not
support the example's `bloomViewGetNativeHandle` call. Official 0.5.1219 supports
that API but its prebuilt standard library fails to link missing HTTP extension
symbols. Official 0.5.1220 had already exposed the same standard-library issue
in the original #153 work. The embedded-view example remains unqualified;
compiler source/build investigation continues rather than removing the example
from the required inventory.

## Hosted Windows execution gap

The #159 Windows shared-test job reported success in about one second. Its
retained log identifies PowerShell as the shell, invokes `ci-check.sh`, then
proceeds to cleanup without any Cargo output or test results. The cache action
also reports that its build paths do not exist. That status does not prove test
execution. Local Windows shared tests were run explicitly through Bash and
Cargo, so their previously published execution evidence remains valid.

The Tests workflow now selects Bash explicitly and requires the emitted shared
test summary on every host and the native-build summary on Windows. Artifact
upload fails when a summary is absent, making another silent non-execution a
failure. The CI command contract checks both the shell and required evidence.
Hosted execution on this correction still needs verification from actual Cargo
output and summary contents.

Audit commands, original failure logs, compiler-release metadata, source patch,
and the six corrected executable hashes are retained under
`tools/quality/out/windows-engine-plan/all-examples/`. The original audit is also
included in the [#159 evidence release](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-ssgi-surface-20260911).

This is progress on #140/#142/#74. All-example PR compilation, real starter and
embedded-view startup, and clean package installation are still required.
