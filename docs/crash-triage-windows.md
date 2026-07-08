# Windows crash triage — native faults in Perry-compiled games

Written during the 2026-07-04 round-2 audit, when the Bloom Shooter hit a
layout-sensitive access violation nobody could reproduce (EN-020). That
bug is now **root-caused and fixed** — see `tickets.md` § EN-020 — and the
hunt left permanent infrastructure behind. This doc describes how to
triage the *next* native fault.

## The engine self-reports now (first stop: stderr)

Since 2026-07-04 (`crash_report` module in `native/windows/src/lib.rs`),
every game linked against the engine:

- catches unhandled SEH exceptions and prints
  `bloom: FATAL unhandled exception 0x<code> at 0x<va> (main.exe+0x<rva>)`
  to stderr, then writes a minidump to `tools/.testout/dumps/`
  (dbghelp is loaded at runtime — its import lib is deliberately NOT on
  perry's link line);
- logs `WM_CLOSE` / `WM_DESTROY` (so a "clean" exit via window death is
  visible — a frozen-looking game whose window died says so);
- logs surface-acquire failures once + every 300th (a game that lost its
  swapchain spins headless: last presented frame stays on screen, input
  looks dead — that's what a user reports as "frozen").

So: **capture stderr** (launch via
`Start-Process -RedirectStandardError`), and the failure class is
usually labeled before you open a debugger.

What still dies silently: fast-fail (`0xC0000409` — Rust abort after a
panic message, or a CRT heap-corruption check) bypasses the filter by
design, and `TerminateProcess`. An empty stderr + no WER event now
narrows to exactly those.

## Reading what it gives you

- **`main.exe+0x<rva>` → function:**
  `llvm-symbolizer --obj=main.exe 0x14000<rva>` (LLVM is at
  `C:\Program Files\LLVM`; add the default PE image base `0x140000000`
  to the RVA; the `--use-native-pdb-reader` flag no longer exists in
  the installed version). Perry-compiled TS functions symbolize to
  `perry_fn_<path>__<name>` with no line info; engine Rust frames get
  file:line from `main.pdb`.
- **Symbols:** `native/windows/Cargo.toml` sets
  `debug = "line-tables-only"`; link games with
  `perry compile src/main.ts -o main --debug-symbols` to get `main.pdb`.
  Keep the exe+pdb pair that produced any dump.
- **Dumps:** open in WinDbg (installed via
  `winget install Microsoft.WinDbg`, launches as `WinDbgX`). Note the
  store WinDbg is GUI-first; for scripted use, pass `-z <dmp>` and mind
  that `-c` command strings containing `;` after `.sympath` get eaten —
  set the sympath interactively or via workspace instead.
- **Do NOT reach for lldb**: the LLVM-bundled `lldb.exe` on the dev box
  delay-load crashes on a missing `python311.dll`.
- **WER LocalDumps** (HKCU, no admin) remains armed as a second net:
  dumps for `main.exe` → `shooter/tools/.testout/dumps/`, error dialog
  suppressed. The WER Application-log event (Id 1000) carries
  module+offset even without a dump:
  `Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000} -MaxEvents 3`.
- **PageHeap** (the definitive overrun-catcher) needs an elevated shell
  for the HKLM IFEO write — the dev shell is not elevated; ask before
  reaching for it.

## Repro harness patterns that actually work (2026-07 hunt)

- Launch the game batch-style with stdout/stderr redirected; the game
  steals the whole screen, so runs are unattended.
- **Keystrokes:** `keybd_event`/SendKeys never reach a
  background-launched game (no keyboard focus). Use
  `PostMessageW(hwnd, WM_KEYDOWN/WM_KEYUP, vk, lparam)` against the
  `FindWindowW(null, 'Bloom Shooter')` handle — the engine's wndproc
  consumes it like a real key.
- **Freeze detection:** pixel-diff consecutive full-frame captures over
  a probe rect. A live frame at TSR 0.5 has ≥ ~2.0 mean-abs-diff of
  grain; exactly 0.0 across 2.5 s means presents stopped. Gate on mean
  luma 15–240 to skip boot black / init-white phases, and make no
  judgment before ~15 s (boot takes 8–12 s and `Process.Responding` is
  false throughout it — it is NOT a hang signal during boot).
- **Exit codes:** set `$proc.EnableRaisingEvents = $true` right after
  `Start-Process` or `.ExitCode` reads back null.
- Reference implementation: the shooter session scratchpad's
  `repro5.ps1`/`repro7.ps1` (title-detect → PostMessage key →
  gameplay soak → per-run stderr/exit-code/dump collection).

## Case study: EN-020 (resolved 2026-07-04)

Three audit-tour crashes at `main.exe+0xe8e5` reading `0x…FFF8` looked
unreproducible — every relink reshuffled the heap and hid it. The
profiler overlay turned out to be a near-deterministic trigger (two
fresh packed-text strings parsed per frame → 6/6 AVs in 7–29 s across
two link layouts, faulting inside `perry_fn_…getProfilerOverlay` /
`…getProfilerFrameHistory`). Root cause: Perry 0.5.x `split()` +
`parseFloat()` read past their own exact-sized slice allocations.
Engine-side tail-padding of `alloc_perry_string` did NOT fix it (the
overread is on Perry-internal allocations) — the fix was the numeric
profiler ABI (`bloom_profiler_row_*` / `_hist_*`), validated 3/3 × 90 s
clean. Moral: when a "random" AV correlates with per-frame string
parsing on Perry, believe the correlation; and a layout-lottery bug
needs a rate amplifier (per-frame allocations), not more fishing runs.
Minidumps from the hunt are archived in
`shooter/tools/.testout/dumps/crash_main_*.dmp`.
