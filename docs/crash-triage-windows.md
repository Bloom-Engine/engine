# Windows crash triage — native AVs in Perry-compiled games

Written after the 2026-07-04 round-2 audit, where the Bloom Shooter hit the
same access violation in three scripted runs (`main.exe+0xe8e5`, read of an
address 8 bytes below a page boundary, exception c0000005) and the fault
could not be reproduced after a relink. Everything below is the kit that
makes the *next* occurrence a five-minute diagnosis instead of a hunt.

## What we know about the 2026-07 AV (EN-020)

- Same faulting instruction offset in all three crashes; faulting address
  `0x…FFF8` = a read that walked off the end of a heap allocation onto an
  unmapped page. Classic buffer-overrun-read signature: the overrun likely
  happens often and only faults when the allocator places the buffer at
  the end of a page — which makes it **layout-sensitive**: any relink
  reshuffles the odds. Our fishing runs on a rebuilt binary (60 s with the
  same workloads, plus a 19-transition feature-toggle gauntlet) did not
  reproduce.
- Crash contexts (shooter audit tour): 20–56 s into runs that combined the
  profiler (enabled), profiler-string FFI reads (`getProfilerOverlay` /
  `getProfilerFrameHistory`), stage transitions, and in two cases runtime
  feature disables. Toggles alone were exonerated by the gauntlet run;
  string churn alone was exonerated by run 3 (crashed with only ~8 tiny
  prints). No Rust panic output on stderr — this is raw UB, not a panic
  (the engine builds with `panic = "abort"`, which would print first).
- Suspect space: Perry runtime heap/string handling at the FFI boundary,
  or an engine-side out-of-bounds read into a heap buffer. The faulting
  module offset (0xe8e5, very low in `.text`) is consistent with a small
  shared helper (memcpy-class) rather than a leaf feature.

## Standing infrastructure on the dev box

- **WER LocalDumps** (HKCU, no admin needed): full dumps for `main.exe`
  land in `shooter/tools/.testout/dumps/`, dialog suppressed
  (`Windows Error Reporting\DontShowUI = 1`). Configured 2026-07-04;
  survives reboots. Every crash from now on leaves a `.dmp`.
- **Symbols**: `native/windows/Cargo.toml` sets
  `[profile.release] debug = "line-tables-only"`, so the staticlib carries
  line tables at negligible cost. Link the game with
  `perry compile src/main.ts -o main --debug-symbols` to get `main.pdb`
  next to the exe (lld `/DEBUG`). Keep the exe+pdb pair that produced any
  dump.
- **Symbolisation** (LLVM is installed at `C:\Program Files\LLVM`):
  `llvm-symbolizer --obj=main.exe --use-native-pdb-reader 0x<RVA>` maps
  the WER event's module offset to a function/line. For a full stack,
  open the `.dmp` in WinDbg (`winget install Microsoft.WinDbg`) with
  `.sympath` pointing at the exe's folder, then `!analyze -v`.
- The WER Application-log event (Id 1000) alone already gives the module
  + offset — check it first:
  `Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000} -MaxEvents 3`.

## Repro harness

The shooter's audit tour (`AUDIT` block for `src/main.ts`, preserved with
the round-2 artifacts) drives the game unattended: scripted combat at the
enemy-pool max, camera pose hops, profiler-string churn every 2 s, and an
optional feature-toggle gauntlet. Historical crash probability was 3/3
within 60 s on the af98dbe-era binary; treat every future tour run as a
fishing run — the dump infrastructure is armed.
