#!/usr/bin/env bash
# Merge libbloom_jolt.a + libJolt.a (built by bloom-shared/build.rs via cmake)
# into libbloom_linux.a. Perry's link step only sees the single staticlib
# named in package.json's perry.nativeLibrary.targets.linux.lib, so the
# C++ Jolt symbols would otherwise be unresolved.
set -euo pipefail

cd "$(dirname "$0")"
TARGET_DIR="target/release"
SRC_LIB="$TARGET_DIR/libbloom_linux.a"
# Output to a separate filename so cargo's hardlink caching doesn't revert
# our merge on the next `cargo build` (cargo restores the cached copy of
# libbloom_linux.a via hardlink). Perry's package.json points at this name.
OUT_LIB="$TARGET_DIR/libbloom_linux_bundled.a"

if [[ ! -f "$SRC_LIB" ]]; then
  echo "bundle-jolt: $SRC_LIB not found — run 'cargo build --release' first" >&2
  exit 1
fi
MAIN_LIB="$SRC_LIB"

JOLT_SHIM=$(find "$TARGET_DIR/build" -name libbloom_jolt.a -path "*/out/lib/*" | head -1)
JOLT_LIB=$(find "$TARGET_DIR/build"  -name libJolt.a       -path "*/out/lib/*" | head -1)

if [[ -z "$JOLT_SHIM" || -z "$JOLT_LIB" ]]; then
  echo "bundle-jolt: could not locate libbloom_jolt.a / libJolt.a under $TARGET_DIR/build" >&2
  exit 1
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Extract each archive into its own subdir so identical object names
# (e.g. bloom_jolt.cpp.o) from different archives don't overwrite each
# other, then rename with a per-archive prefix before merging.
extract() {
  local archive="$1" prefix="$2"
  local subdir="$WORK/$prefix"
  mkdir -p "$subdir"
  ( cd "$subdir" && ar x "$archive" )
  for f in "$subdir"/*.o; do
    mv "$f" "$WORK/${prefix}_$(basename "$f")"
  done
  rmdir "$subdir"
}

extract "$(realpath "$MAIN_LIB")"  bloom
extract "$(realpath "$JOLT_SHIM")" jshim
extract "$(realpath "$JOLT_LIB")"  jolt

# Build the merged staticlib. Output to OUT_LIB so cargo doesn't clobber it
# via its hardlink cache.
rm -f "$OUT_LIB"
ar crs "$OUT_LIB" "$WORK"/*.o

echo "bundle-jolt: merged $(ls "$WORK"/*.o | wc -l) object files into $OUT_LIB"
