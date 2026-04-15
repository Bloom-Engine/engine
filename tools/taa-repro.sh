#!/usr/bin/env bash
# Build, run renderer-test in interactive-capture mode, and detect
# whether the TAA+bloom corruption reproduces. Exit 0 = clean, 1 = corrupted.
#
# Usage:  tools/taa-repro.sh [frames=120] [out=/tmp/taa-repro.png]

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMES="${1:-120}"
OUT="${2:-/tmp/taa-repro.png}"

(cd "$REPO/native/macos" && cargo build --release >/dev/null 2>&1)
(cd "$REPO/examples/renderer-test" && perry compile main.ts >/dev/null 2>&1)
(cd "$REPO/examples/renderer-test" && ./main --interactive-capture "$FRAMES" "$OUT" >/dev/null 2>&1)

"$REPO/tools/detect-corruption.sh" "$OUT"
