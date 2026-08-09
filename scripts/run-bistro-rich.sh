#!/usr/bin/env bash
# Build and launch the richest Bistro profile that the current non-instanced
# ModelData ABI can load safely. It retains all 551 unique source meshes at
# their first authored transform without expanding 2,909 nodes to ~19 GB.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXAMPLE="$ROOT/examples/bistro"
OUTPUT="$EXAMPLE/.generated/bistro-rich.gltf"

mkdir -p "$(dirname "$OUTPUT")"
python3 "$ROOT/tools/quality/prepare_bistro.py" \
  "$EXAMPLE/assets/bistro.gltf" \
  "$OUTPUT" \
  --metadata "$EXAMPLE/.generated/bistro-rich.json" \
  --selection all-unique \
  --revision 7c9f9f9ac0915024ccf3dddbccd8bfc643a42607
python3 "$ROOT/tools/quality/build_example.py" "$EXAMPLE"

cd "$EXAMPLE"
exec ./main --scene .generated/bistro-rich.gltf "$@"
