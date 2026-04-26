#!/usr/bin/env bash
# Local parity for .github/workflows/test.yml.
#
# Runs the subset of CI checks that make sense on a dev machine:
#   - bloom-shared unit tests (includes Jolt C++ shim + jolt_sys tests)
#   - the *current host's* platform crate build (macOS / Linux / Windows)
#   - cargo check for shared on wasm32 (skips the full wasm-pack build)
#
# Usage:
#   ./scripts/ci-check.sh              # run the full local suite
#   ./scripts/ci-check.sh --fast       # skip the host platform crate build
#   ./scripts/ci-check.sh --wasm       # also run wasm-pack build (slow, needs wasm-pack)

set -euo pipefail

FAST=0
INCLUDE_WASM=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    --wasm) INCLUDE_WASM=1 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

host_os="$(uname -s)"
case "$host_os" in
  Darwin)  host_crate="macos" ;;
  Linux)   host_crate="linux" ;;
  MINGW*|MSYS*|CYGWIN*) host_crate="windows" ;;
  *) host_crate="" ;;
esac

hr() { printf '\n==> %s\n' "$*"; }

hr "bloom-shared: cargo test --release"
( cd native/shared && cargo test --release )

hr "bloom-shared: cargo check (wasm32, web feature)"
( cd native/shared && cargo check --target wasm32-unknown-unknown --no-default-features --features web )

if [ "$FAST" -eq 0 ] && [ -n "$host_crate" ]; then
  hr "bloom-$host_crate: cargo build --release"
  ( cd "native/$host_crate" && cargo build --release )
fi

if [ "$INCLUDE_WASM" -eq 1 ]; then
  if ! command -v wasm-pack >/dev/null 2>&1; then
    echo "wasm-pack not installed — skipping wasm-pack build" >&2
  else
    hr "bloom-web: wasm-pack build --release --target web"
    ( cd native/web && wasm-pack build --release --target web )
  fi
fi

hr "OK — all local checks passed"
