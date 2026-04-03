#!/bin/bash
# Build Bloom Engine for Web
#
# Usage:
#   ./native/web/build.sh [game.ts] [--output dist/]
#
# Steps:
#   1. Build bloom_web.wasm via wasm-pack
#   2. Compile game TypeScript via perry --target wasm (if provided)
#   3. Assemble output directory with all files needed to serve
#
# Prerequisites:
#   - wasm-pack: cargo install wasm-pack
#   - perry: ../../../perry/perry/target/release/perry (or in PATH)
#   - wasm-opt (optional): for binary size optimization

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEB_CRATE="$SCRIPT_DIR"
OUTPUT_DIR="${2:-$ENGINE_DIR/dist/web}"
GAME_FILE="$1"

echo "=== Bloom Web Build ==="
echo ""

# 1. Build Bloom WASM via wasm-pack
echo "[1/4] Building bloom_web.wasm..."
cd "$WEB_CRATE"
wasm-pack build --target web --out-dir pkg --no-typescript 2>&1 | tail -3
echo "  Output: $WEB_CRATE/pkg/"

# 2. Optimize WASM binary (if wasm-opt is available)
if command -v wasm-opt &> /dev/null; then
  echo "[2/4] Optimizing WASM with wasm-opt..."
  WASM_FILE="$WEB_CRATE/pkg/bloom_web_bg.wasm"
  ORIG_SIZE=$(wc -c < "$WASM_FILE")
  wasm-opt -Oz "$WASM_FILE" -o "$WASM_FILE.opt"
  mv "$WASM_FILE.opt" "$WASM_FILE"
  OPT_SIZE=$(wc -c < "$WASM_FILE")
  echo "  Optimized: $((ORIG_SIZE / 1024))KB → $((OPT_SIZE / 1024))KB"
else
  echo "[2/4] Skipping wasm-opt (not installed). Install with: cargo install wasm-opt"
fi

# 3. Compile game (if provided)
if [ -n "$GAME_FILE" ] && [ -f "$GAME_FILE" ]; then
  echo "[3/4] Compiling game: $GAME_FILE"

  # Find perry compiler
  PERRY=""
  if command -v perry &> /dev/null; then
    PERRY="perry"
  elif [ -f "$ENGINE_DIR/../../perry/perry/target/release/perry" ]; then
    PERRY="$ENGINE_DIR/../../perry/perry/target/release/perry"
  else
    echo "  ERROR: perry compiler not found. Install it or add to PATH."
    exit 1
  fi

  $PERRY "$GAME_FILE" --target wasm -o "/tmp/bloom_game.html"
  echo "  Game compiled to WASM"
else
  echo "[3/4] No game file specified, skipping game compilation"
fi

# 4. Assemble output directory
echo "[4/4] Assembling output..."
mkdir -p "$OUTPUT_DIR"

# Copy Bloom WASM package
cp -r "$WEB_CRATE/pkg" "$OUTPUT_DIR/pkg"

# Copy HTML template and glue
cp "$WEB_CRATE/index.html" "$OUTPUT_DIR/index.html"
cp "$WEB_CRATE/bloom_glue.js" "$OUTPUT_DIR/bloom_glue.js"

# Copy game assets (if game directory has an assets/ folder)
if [ -n "$GAME_FILE" ]; then
  GAME_DIR="$(dirname "$(realpath "$GAME_FILE")")"
  if [ -d "$GAME_DIR/assets" ]; then
    cp -r "$GAME_DIR/assets" "$OUTPUT_DIR/assets"
    echo "  Copied assets/ directory"
  fi
fi

# Calculate total size
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
WASM_SIZE=$(wc -c < "$OUTPUT_DIR/pkg/bloom_web_bg.wasm" 2>/dev/null || echo "0")

echo ""
echo "=== Build Complete ==="
echo "  Output: $OUTPUT_DIR"
echo "  WASM size: $((WASM_SIZE / 1024))KB"
echo "  Total size: $TOTAL_SIZE"
echo ""
echo "To serve locally:"
echo "  cd $OUTPUT_DIR && python3 -m http.server 8080"
echo "  Open http://localhost:8080"
