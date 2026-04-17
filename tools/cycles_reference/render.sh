#!/usr/bin/env bash
# Path-traced Cycles reference render of the Bloom Bistro scene.
#
# Usage:
#   ./render.sh                       # 128 samples -> /tmp/bistro_cycles.png
#   ./render.sh --samples 512
#   ./render.sh --out /tmp/foo.png --samples 256 --device CPU
#
# Requires Blender 5.0+ on PATH (macOS: `brew install --cask blender` or
# download from https://www.blender.org/download/).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$HERE/render.py"

if ! command -v blender >/dev/null 2>&1; then
  echo "[cycles_reference] ERROR: 'blender' not found on PATH." >&2
  echo "  Install from https://www.blender.org/download/ (or 'brew install --cask blender' on macOS)." >&2
  exit 127
fi

echo "[cycles_reference] blender: $(command -v blender)"
blender --version | head -1

# Pass everything after `--` through to the Python script.
exec blender -b -P "$PY_SCRIPT" -- "$@"
