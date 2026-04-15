#!/usr/bin/env bash
# Corruption detector for interactive-capture screenshots.
#
# Heuristic: a clean scene has ≲1% pure-black pixels; the "big black
# bars" corruption pushes that >15%.
#
# Usage:  tools/detect-corruption.sh path.png [threshold_percent]
# Exit 0 = clean, 1 = corrupted.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 IMAGE.png [threshold_percent=15]" >&2
  exit 2
fi

IMG="$1"
THRESHOLD_PCT="${2:-15}"

if [[ ! -f "$IMG" ]]; then
  echo "missing: $IMG" >&2
  exit 2
fi

python3 - "$IMG" "$THRESHOLD_PCT" <<'PY'
import sys
from PIL import Image

img_path, threshold_str = sys.argv[1], sys.argv[2]
threshold_pct = float(threshold_str)

with Image.open(img_path) as im:
    rgb = im.convert("RGB")
    pixels = rgb.tobytes()
    total = len(pixels) // 3
    # Count pure-black pixels (R==G==B==0). Fast path via bytes scan.
    black = sum(1 for i in range(0, len(pixels), 3)
                if pixels[i] == 0 and pixels[i+1] == 0 and pixels[i+2] == 0)

pct = 100.0 * black / total
print(f"black-pixel fraction: {pct:.2f}% (threshold {threshold_pct}%)")
if pct > threshold_pct:
    print("CORRUPTED")
    sys.exit(1)
else:
    print("CLEAN")
    sys.exit(0)
PY
