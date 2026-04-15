#!/usr/bin/env bash
# Multi-camera validation: render the same scene from N viewpoints with
# both bloom-reference (CPU path tracer) and bloom-renderer (realtime),
# diff each pair, print per-view + aggregate metrics.
#
# Usage: tools/validate.sh [--width W --height H --spp N --bounces N]
#
# Cameras live in this script, not a JSON file — we want them in source
# control next to the metric script that consumes them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RENDERER_TEST="$REPO_ROOT/examples/renderer-test"
REF_BIN="$REPO_ROOT/tools/bloom-reference/target/release/bloom-reference"
DIFF_BIN="$REPO_ROOT/tools/bloom-diff/target/release/bloom-diff"
RT_BIN="$RENDERER_TEST/main"
SPEC="$RENDERER_TEST/specs/helmet.json"
OUT_DIR="$REPO_ROOT/tools/validate-out"

WIDTH=512
HEIGHT=512
SPP=64
BOUNCES=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --width)   WIDTH="$2";   shift 2 ;;
    --height)  HEIGHT="$2";  shift 2 ;;
    --spp)     SPP="$2";     shift 2 ;;
    --bounces) BOUNCES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Camera viewpoints: name | px py pz | tx ty tz | fov_y_deg
# Picked to exercise different parts of the helmet:
#   front      — straight-on visor (specular highlights, normal map detail)
#   three-quarter — original spec angle (overall PBR check)
#   side       — profile silhouette (rim light, edge sampling)
#   top-down   — env reflection on top hemisphere (IBL coverage)
CAMERAS=(
  "front      0.0 0.5 3.0   0 0 0   45"
  "threequarter 1.8 1.2 2.4   0 0 0   45"
  "side       3.0 0.4 0.0   0 0 0   45"
  "topdown    0.0 3.0 0.5   0 0 0   45"
)

mkdir -p "$OUT_DIR"

# Sanity-check binaries.
for bin in "$REF_BIN" "$DIFF_BIN" "$RT_BIN"; do
  if [[ ! -x "$bin" ]]; then
    echo "missing or not executable: $bin" >&2
    echo "build it first (see tools/README.md)" >&2
    exit 1
  fi
done

printf '%-12s  %8s  %8s  %8s\n' "view" "RMSE" "SSIM" "%>tol"
printf '%-12s  %8s  %8s  %8s\n' "----" "----" "----" "-----"

total_rmse=0
total_ssim=0
total_tol=0
n=0

for entry in "${CAMERAS[@]}"; do
  name=$(echo "$entry" | awk '{print $1}')
  px=$(echo "$entry" | awk '{print $2}'); py=$(echo "$entry" | awk '{print $3}'); pz=$(echo "$entry" | awk '{print $4}')
  tx=$(echo "$entry" | awk '{print $5}'); ty=$(echo "$entry" | awk '{print $6}'); tz=$(echo "$entry" | awk '{print $7}')
  fov=$(echo "$entry" | awk '{print $8}')

  ref="$OUT_DIR/ref-$name.png"
  rt="$OUT_DIR/rt-$name.png"
  composite="$OUT_DIR/diff-$name.png"

  # Reference render — silenced; rerun by deleting its output.
  if [[ ! -f "$ref" ]]; then
    "$REF_BIN" \
      --spec "$SPEC" \
      --camera "$px" "$py" "$pz" "$tx" "$ty" "$tz" "$fov" \
      --width "$WIDTH" --height "$HEIGHT" \
      --spp "$SPP" --bounces "$BOUNCES" \
      --out "$ref" >/dev/null
  fi

  # Realtime render at native window size (Retina-doubled).
  rt_native="$OUT_DIR/rt-$name-native.png"
  ( cd "$RENDERER_TEST" && ./main \
      --camera "$px" "$py" "$pz" "$tx" "$ty" "$tz" "$fov" \
      --out "$rt_native" ) >/dev/null

  # Read the actual native dimensions and downscale to WIDTH×HEIGHT
  # using sips (macOS native; no extra install required) so the diff
  # works against the small reference render. Keeps the reference
  # cheap (low spp at low res) without forcing the realtime to fight
  # macOS's Retina upscaling.
  sips -z "$HEIGHT" "$WIDTH" "$rt_native" --out "$rt" >/dev/null 2>&1

  # Diff. Allow the binary to exit non-zero (it does so when above
  # tolerance) — we just want the metric output.
  metrics=$("$DIFF_BIN" \
    --reference "$ref" \
    --candidate "$rt" \
    --composite "$composite" \
    --tolerance 0.05 2>&1 || true)

  rmse=$(echo "$metrics" | awk '/RMSE \(luminance\)/ {print $3}')
  ssim=$(echo "$metrics" | awk '/SSIM \(luminance\)/ {print $3}')
  tol=$(echo  "$metrics" | awk '/% above tolerance/ {print $4}')

  printf '%-12s  %8s  %8s  %8s\n' "$name" "$rmse" "$ssim" "$tol"

  # Accumulate using awk (bash floats are awkward).
  total_rmse=$(awk -v a="$total_rmse" -v b="$rmse" 'BEGIN{print a+b}')
  total_ssim=$(awk -v a="$total_ssim" -v b="$ssim" 'BEGIN{print a+b}')
  total_tol=$(awk  -v a="$total_tol"  -v b="${tol%\%}" 'BEGIN{print a+b}')
  n=$((n+1))
done

avg_rmse=$(awk -v t="$total_rmse" -v n="$n" 'BEGIN{printf "%.5f", t/n}')
avg_ssim=$(awk -v t="$total_ssim" -v n="$n" 'BEGIN{printf "%.5f", t/n}')
avg_tol=$(awk  -v t="$total_tol"  -v n="$n" 'BEGIN{printf "%.2f%%", t/n}')

printf '%-12s  %8s  %8s  %8s\n' "----" "----" "----" "-----"
printf '%-12s  %8s  %8s  %8s\n' "average" "$avg_rmse" "$avg_ssim" "$avg_tol"
