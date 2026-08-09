#!/usr/bin/env bash
# Build and launch the complete authored Bistro scene. Bloom retains one
# immutable payload for each source primitive and submits all 2,909 placements,
# so the full asset no longer needs a generated unique-mesh substitute.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXAMPLE="$ROOT/examples/bistro"
python3 "$ROOT/tools/quality/build_example.py" "$EXAMPLE"

cd "$EXAMPLE"
exec ./main --scene assets/bistro.gltf "$@"
