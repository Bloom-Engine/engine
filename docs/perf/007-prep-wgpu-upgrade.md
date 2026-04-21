# 007-prep — wgpu upgrade (Metal ray-query unlock)

**Effort:** 2-3 days · **Expected gain:** none (enabler) · **Status:** open

## Problem

Bloom pins `wgpu = "24"` in all 8 wgpu-using crates (shared, macos, ios, tvos,
windows, linux, android, web). wgpu 24 lacks Metal ray-query support — it was
added post-v24 via upstream PRs #7402 and #8071, released in v25/v26. Phase 1
of the Lumen GI rollout (ticket 007b) needs `Features::EXPERIMENTAL_RAY_QUERY`
on Metal to ship HW ray-tracing on macOS / iOS / tvOS. Without this bump, the
HW path is Windows+Linux only and Apple platforms stall on SW.

This ticket is the serial prerequisite for 007a and 007b.

## Approach

Bump to the current stable wgpu release that contains Metal ray-query. Target
the highest released version whose `CHANGELOG.md` confirms Metal RT (likely 26.x
at the time this ticket starts — verify at kickoff). Sweep API churn across:

- `native/shared/Cargo.toml` — base pin.
- `native/{macos,ios,tvos,windows,linux,android,web}/Cargo.toml` — per-platform pins.
- All 8 `Cargo.lock` files.
- `native/shared/src/renderer/mod.rs` — largest surface area; will touch most
  breaking changes (descriptor field renames, feature-flag bits, surface-config
  shapes, ray-tracing types if they became stable).
- `native/shared/src/renderer/shaders.rs` — naga WGSL parser accepts a slightly
  different shape each release (e.g. attribute syntax, `override` semantics).
- Platform crates — window creation / surface config wiring.

No renderer logic changes beyond what the compiler forces. No Lumen code yet.

## Acceptance

- `cd native/shared && cargo check` clean (default features).
- `cd native/shared && cargo check --target wasm32-unknown-unknown --no-default-features --features web` clean.
- `cd native/macos && cargo build --release` clean.
- `cd native/web && cargo check --target wasm32-unknown-unknown` clean.
- `cd native/windows && cargo check --target x86_64-pc-windows-msvc` clean
  (if cross-compile toolchain present; otherwise document in commit that
  the tree compiles and leave Windows validation for a Windows host).
- `./examples/intel-sponza/main --quality 3 --fps-only 300` within ±2% of the
  pre-upgrade fps number. Any regression is fixed before commit, not merged.
- `./examples/intel-sponza/main --quality 0 --fps-only 60` hits 60 fps.
- `./examples/intel-sponza/main --capture 30 /tmp/sponza_007prep_after.png` —
  pixel-identical (or within TAA noise floor) against `/tmp/sponza_baseline.png`.
- Adapter dump confirms `Features::EXPERIMENTAL_RAY_QUERY` is reported on the
  Apple Silicon Metal adapter. Record in commit message.

## Files likely to change

- 8 `Cargo.toml` files (native/shared + 7 platform crates).
- 8 `Cargo.lock` files.
- `native/shared/src/renderer/mod.rs` — descriptor / feature / surface churn.
- `native/shared/src/renderer/shaders.rs` — WGSL parser churn if any.
- `native/web/src/lib.rs` — `wasm-bindgen` interactions that touch wgpu types.
- Platform `lib.rs` / `main.rs` files where surface creation happens.

## Notes

- Don't touch `raw-window-handle`'s pin unless the new wgpu requires it.
- If the upgrade pulls in a newer `naga`, run `tools/` shader dumps (if any)
  as smoke tests. The existing SSGI / SSAO / main_hdr shaders are the ones
  most likely to trip a parser change.
- Two-major-version hop — do not attempt to combine with any other perf ticket.
